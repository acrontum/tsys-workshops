"""Check graph reuse against a robot controller with an analytic solution."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "Requires CUDA PyTorch")
class PlaybackControllerTests(unittest.TestCase):
    @staticmethod
    def pose_error(**inputs):
        return (
            inputs["ctrl_target_fingertip_midpoint_pos"] - inputs["fingertip_midpoint_pos"],
            torch.zeros_like(inputs["fingertip_midpoint_pos"]),
        )

    def test_graph_updates_inputs_and_preserves_nullspace_and_torque_limits(self):
        spec = importlib.util.spec_from_file_location(
            "playback_controller", ROOT / "example2_parallel_rl/scripts/playback_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for count in (1, 4, 8):
            with self.subTest(environments=count):
                cfg = SimpleNamespace(
                    scene=SimpleNamespace(num_envs=count),
                    ctrl=SimpleNamespace(default_dof_pos_tensor=[0.0]*7, kp_null=10.0, kd_null=6.3246),
                )
                zero3 = torch.zeros(count, 3, device="cuda")
                quaternion = torch.zeros(count, 4, device="cuda")
                quaternion[:, 3] = 1
                joints = torch.zeros(count, 9, device="cuda")
                joints[:, 6] = 0.1
                velocity = torch.zeros_like(joints)
                velocity[:, 6] = 0.05
                jacobian = torch.zeros(count, 6, 7, device="cuda")
                jacobian[:, :, :6] = torch.eye(6, device="cuda")
                inputs = dict(
                    cfg=cfg, device="cuda", dof_pos=joints, dof_vel=velocity,
                    fingertip_midpoint_pos=zero3, fingertip_midpoint_quat=quaternion,
                    fingertip_midpoint_linvel=zero3, fingertip_midpoint_angvel=zero3,
                    jacobian=jacobian, arm_mass_matrix=torch.eye(7, device="cuda").repeat(count, 1, 1),
                    ctrl_target_fingertip_midpoint_pos=zero3.clone(),
                    ctrl_target_fingertip_midpoint_quat=quaternion,
                    task_prop_gains=torch.full((count, 6), 100.0, device="cuda"),
                    task_deriv_gains=torch.ones(count, 6, device="cuda"),
                    dead_zone_thresholds=None,
                )
                controller = module.PlaybackController(self.pose_error)
                for target in (0.0, 0.2, 2.0, -2.0):
                    inputs["ctrl_target_fingertip_midpoint_pos"].fill_(target)
                    torque, _ = controller(**inputs)
                    controller.check()
                    expected = torch.zeros_like(joints)
                    expected[:, :3] = max(-100.0, min(100.0, target * 100.0))
                    expected[:, 6] = -1.0 - 6.3246 * 0.05
                    torch.testing.assert_close(torque, expected, atol=3e-6, rtol=3e-6)
                    if target == 0.0:
                        graph = controller.graph
                    self.assertIs(controller.graph, graph)
                # A singular matrix must not inject NaN/Inf torques into PhysX.
                inputs["arm_mass_matrix"].zero_()
                torque, _ = controller(**inputs)
                self.assertTrue(bool(torch.isfinite(torque).all()))
                with self.assertRaisesRegex(RuntimeError, "singular matrix"):
                    controller.check()


if __name__ == "__main__":
    unittest.main()
