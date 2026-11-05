"""Reuse Factory's operational-space controller as one CUDA graph for playback.

The controller equations, gains and torque limits match the installed Factory
controller. Constant tensors and GPU work are reused between physics steps.
"""

import math

import torch


class PlaybackController:
    def __init__(self, pose_error, math_utils=None):
        self.pose_error = pose_error
        self.math_utils = math_utils
        self.graph = None

    def _targets(self, inputs):
        cfg = inputs["cfg"]
        position = inputs["fingertip_midpoint_pos"] + inputs["actions"][:, :3] * inputs["pos_threshold"]
        frame = inputs["fixed_pos_obs_frame"] + inputs["init_fixed_pos_obs_noise"]
        position = frame + torch.clip(
            position - frame, -cfg.ctrl.pos_action_bounds[0], cfg.ctrl.pos_action_bounds[1]
        )
        rotation = inputs["actions"][:, 3:6] * inputs["rot_threshold"]
        angle = torch.linalg.vector_norm(rotation, dim=-1)
        axis = rotation / angle.unsqueeze(-1)
        quaternion = self.math_utils.quat_from_angle_axis(angle, axis)
        quaternion = torch.where(
            angle.unsqueeze(-1) > 1.0e-6, quaternion, self.identity_quaternion
        )
        quaternion = self.math_utils.quat_mul(quaternion, inputs["fingertip_midpoint_quat"])
        euler = torch.stack(self.math_utils.euler_xyz_from_quat(quaternion), dim=1)
        euler[:, 0] = 3.14159
        euler[:, 1] = 0.0
        quaternion = self.math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        return position, quaternion

    def _calculate(self, inputs):
        cfg = inputs["cfg"]
        if self.math_utils is None:
            target_position = inputs["ctrl_target_fingertip_midpoint_pos"]
            target_quaternion = inputs["ctrl_target_fingertip_midpoint_quat"]
        else:
            target_position, target_quaternion = self._targets(inputs)
        position_error, rotation_error = self.pose_error(
            fingertip_midpoint_pos=inputs["fingertip_midpoint_pos"],
            fingertip_midpoint_quat=inputs["fingertip_midpoint_quat"],
            ctrl_target_fingertip_midpoint_pos=target_position,
            ctrl_target_fingertip_midpoint_quat=target_quaternion,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        error = torch.cat((position_error, rotation_error), dim=1)
        velocity = torch.cat(
            (inputs["fingertip_midpoint_linvel"], inputs["fingertip_midpoint_angvel"]), dim=1
        )
        wrench = inputs["task_prop_gains"] * error - inputs["task_deriv_gains"] * velocity
        dead_zone = inputs.get("dead_zone_thresholds")
        if dead_zone is not None:
            wrench = torch.where(
                wrench.abs() < dead_zone, torch.zeros_like(wrench),
                wrench.sign() * (wrench.abs() - dead_zone),
            )
        jacobian = inputs["jacobian"]
        transpose = jacobian.transpose(1, 2)
        mass = inputs["arm_mass_matrix"]
        mass_inverse, mass_info = torch.linalg.inv_ex(mass, check_errors=False)
        task_inverse, task_info = torch.linalg.inv_ex(
            jacobian @ mass_inverse @ transpose, check_errors=False
        )
        inverse_jacobian = task_inverse @ jacobian @ mass_inverse
        displacement = self.default_joints - inputs["dof_pos"][:, :7]
        displacement = (displacement + math.pi) % (2 * math.pi) - math.pi
        null_command = cfg.ctrl.kd_null * -inputs["dof_vel"][:, :7] + cfg.ctrl.kp_null * displacement
        null_torque = (
            self.identity - transpose @ inverse_jacobian
        ) @ (mass @ null_command.unsqueeze(-1))
        arm_torque = (transpose @ wrench.unsqueeze(-1)) + null_torque
        torque = torch.zeros_like(inputs["dof_pos"])
        torque[:, :7] = arm_torque.squeeze(-1)
        torque = torque.clamp(-100.0, 100.0)
        valid = (mass_info == 0) & (task_info == 0) & torch.isfinite(torque).all(dim=1)
        # Do not feed invalid torques to PhysX. The host check reports the error
        # before the next policy action, rather than synchronizing eight times.
        torque = torch.where(valid[:, None], torque, torch.zeros_like(torque))
        return torque, wrench, valid

    def __call__(self, **inputs):
        if self.graph is None:
            device = inputs["dof_pos"].device
            cfg = inputs["cfg"]
            self.default_joints = torch.tensor(
                cfg.ctrl.default_dof_pos_tensor, device=device
            ).repeat(cfg.scene.num_envs, 1)
            self.identity = torch.eye(7, device=device).unsqueeze(0)
            self.identity_quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
            self.inputs = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self._calculate(self.inputs)
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
                self.outputs = self._calculate(self.inputs)
            print("PLAYBACK: GPU arm controller ready.", flush=True)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                self.inputs[key].copy_(value)
        self.graph.replay()
        return self.outputs[:2]

    def check(self):
        if self.graph is not None and not bool(self.outputs[2].all()):
            raise RuntimeError("Playback controller encountered a singular matrix or non-finite torque.")

    def apply(self, env):
        """Apply one GearMesh physics update using the original Factory targets."""
        if env.cfg_task.unidirectional_rot:
            raise RuntimeError("This playback controller supports the GearMesh task only.")
        if env.last_update_timestamp < env._robot._data._sim_timestamp:
            env._compute_intermediate_values(dt=env.physics_dt)
        env.joint_torque, env.applied_wrench = self(
            cfg=env.cfg,
            dof_pos=env.joint_pos,
            dof_vel=env.joint_vel,
            fingertip_midpoint_pos=env.fingertip_midpoint_pos,
            fingertip_midpoint_quat=env.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=env.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=env.fingertip_midpoint_angvel,
            jacobian=env.fingertip_midpoint_jacobian,
            arm_mass_matrix=env.arm_mass_matrix,
            task_prop_gains=env.task_prop_gains,
            task_deriv_gains=env.task_deriv_gains,
            actions=env.actions,
            pos_threshold=env.pos_threshold,
            rot_threshold=env.rot_threshold,
            fixed_pos_obs_frame=env.fixed_pos_obs_frame,
            init_fixed_pos_obs_noise=env.init_fixed_pos_obs_noise,
            dead_zone_thresholds=env.dead_zone_thresholds,
            device=env.device,
        )
        env.ctrl_target_joint_pos[:, 7:9] = 0.0
        env.joint_torque[:, 7:9] = 0.0
        env._robot.set_joint_position_target_index(target=env.ctrl_target_joint_pos)
        env._robot.set_joint_effort_target_index(target=env.joint_torque)
