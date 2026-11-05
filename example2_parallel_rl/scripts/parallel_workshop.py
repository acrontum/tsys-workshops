"""Visible randomized Factory insertion, batched Gym steps and real PPO updates."""
import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import warp as wp
wp.config.enable_backward = False
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["resets", "train", "evaluate", "deploy", "snapshot"], default="resets")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--iterations", type=int, default=12)
p.add_argument("--rollout", type=int, default=32)
p.add_argument("--episodes", type=int, default=2)
p.add_argument("--resets", type=int, default=3)
p.add_argument("--hold_seconds", type=float, default=10)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--checkpoint", default="")
p.add_argument("--from_scratch", action="store_true")
p.add_argument("--output_dir", type=Path, default=None, help="Override the generated-results folder.")
p.add_argument("--render_quality", choices=["fast", "standard"], default="fast",
               help="Fast live rendering, or the original renderer settings.")
p.add_argument("--render_every", type=int, default=0,
               help="Render once per N policy steps; 0 selects the quality preset.")
p.add_argument("--playback_mode", choices=["fast", "reference"], default="fast",
               help="For deploy: reuse GPU controller work and skip training rewards, or use the original loop.")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.render_every < 0:
    p.error("--render_every must be zero or positive.")
if a.mode=="train" and (a.rollout % 32 or a.num_envs*a.rollout < 128):
    p.error("Use a rollout divisible by 32 and at least 128 batched transitions.")
ROOT = Path(__file__).resolve().parents[1]
OUT = a.output_dir if a.output_dir is not None else ROOT/"output"
OUT.mkdir(parents=True, exist_ok=True)
if a.mode in ("train", "evaluate", "deploy"):
    report_name = "training_report.json" if a.mode == "train" else a.mode + "_report.json"
    (OUT / report_name).unlink(missing_ok=True)
gui = "kit" in (a.visualizer or [])
if gui:
    import importlib.util
    if not a.experience:
        a.experience = str(Path(importlib.util.find_spec("isaacsim").origin).parent/"apps/isaacsim.exp.base.python.kit")
    a.kit_args = (a.kit_args or "")+" --/app/window/width=1360 --/app/window/height=800"
app = AppLauncher(a).app
import carb.settings
render_settings = carb.settings.get_settings()
if gui:
    print("ORIGINAL RTX MODE:", render_settings.get("/rtx/rendermode"), flush=True)
    if a.render_quality == "fast":
        # Live observation uses simple real-time shading. Dataset generation is
        # a separate process and retains its own camera/render settings.
        for name, value in {
            "/rtx/rendermode": "MinimalRendering",
            "/rtx/minimal/mode": 2,
            "/rtx/reflections/enabled": False,
            "/rtx/indirectDiffuse/enabled": False,
            "/rtx/translucency/enabled": False,
            "/rtx/shadows/enabled": False,
            "/rtx/ambientOcclusion/enabled": False,
            "/rtx/directLighting/sampledLighting/samplesPerPixel": 1,
            "/rtx-transient/dlssg/enabled": False,
        }.items():
            render_settings.set(name, value)

import gymnasium as gym
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, register_rl_games_env
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
from isaaclab_tasks.contrib.factory.factory_env import FactoryEnv
from isaaclab_tasks.contrib.factory import factory_utils
from workbench import add_nist_bench

warnings.filterwarnings("ignore", category=DeprecationWarning)
cfg = parse_env_cfg("IsaacContrib-Factory-GearMesh-Direct", device=a.device, num_envs=a.num_envs)
cfg.seed = a.seed
fast_playback = a.mode == "deploy" and a.playback_mode == "fast"
cfg.sim.physics.gpu_collision_stack_size = 2**27
cfg.sim.physics.gpu_max_rigid_contact_count = 2**21
cfg.sim.physics.gpu_max_rigid_patch_count = 2**20
render_every = a.render_every or (1 if fast_playback else (2 if a.render_quality == "fast" else 1))
cfg.sim.render_interval = cfg.decimation * render_every
if a.mode in ("resets","snapshot"):
    cfg.scene.clone_in_fabric=False
add_nist_bench(cfg)
if fast_playback:
    import isaaclab.sim as sim_utils
    # The rear board is a visual reference, not part of the insertion contacts.
    cfg.scene.nist_board.spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)
    cfg.scene.nist_board.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
# These are the validated Factory distributions, made explicit for the lesson.
cfg.task.fixed_asset_init_pos_noise = [0.05, 0.05, 0.05]
cfg.task.fixed_asset_init_orn_deg = 0.0
cfg.task.fixed_asset_init_orn_range_deg = 15.0
cfg.task.hand_init_pos_noise = [0.02, 0.02, 0.01]
cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.785]
cfg.task.held_asset_pos_noise = [0.003, 0.0, 0.003]
eye = (6.5, -7.5, 6.8) if a.num_envs > 1 else (0.9, -0.48, 0.40)
target = (0.4, 0.0, 0.15) if a.num_envs > 1 else (0.60, 0.015, 0.075)
cfg.viewer.eye, cfg.viewer.lookat = eye, target
if gui:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=eye, lookat=target)]

# Match the published network's WXYZ inputs; leave physics in native XYZW.
original_dict = FactoryEnv._get_factory_obs_state_dict
def adapted_observations(self):
    obs, states = original_dict(self)
    for values in (obs, states):
        for key in values:
            if "quat" in key:
                values[key] = values[key][:, [3,0,1,2]]
    return obs, states
FactoryEnv._get_factory_obs_state_dict = adapted_observations

state = dict(mode=a.mode, phase="Loading workcells", num_envs=a.num_envs,
             epoch=0, epochs=a.iterations, policy_steps=0, transitions=0,
             updates=[], resets=0, last_reward=None, total_successes=0, total_trials=0,
             starts_from="Untrained network" if a.from_scratch else "Published trained policy")
def publish(**extra):
    state.update(extra)
    temp = OUT/"live_state.tmp"
    temp.write_text(json.dumps(state, indent=2, allow_nan=False))
    temp.replace(OUT/"live_state.json")
publish()
(OUT/"reset_samples.jsonl").write_text("")
reset_count = 0
original_randomize = FactoryEnv.randomize_initial_state
def observed_reset(self, env_ids):
    global reset_count
    result = original_randomize(self, env_ids)
    reset_count += 1
    rows = []
    for idx in env_ids.tolist():
        rows.append(dict(environment=idx+1, fixed_position=self.fixed_pos[idx].tolist(),
                         held_position=self.held_pos[idx].tolist(),
                         hand_position=self.fingertip_midpoint_pos[idx].tolist(),
                         fixed_quaternion_xyzw=self.fixed_quat[idx].tolist(),
                         hand_quaternion_xyzw=self.fingertip_midpoint_quat[idx].tolist()))
    sample = dict(reset=reset_count, seed=a.seed, environments=rows)
    with (OUT/"reset_samples.jsonl").open("a") as f:
        f.write(json.dumps(sample)+"\n")
    print("RANDOMIZED RESET", reset_count, json.dumps(rows), flush=True)
    publish(resets=reset_count, randomized_starts=rows, phase="New randomized starting poses")
    return result
FactoryEnv.randomize_initial_state = observed_reset
env = gym.make("IsaacContrib-Factory-GearMesh-Direct", cfg=cfg)
base = env.unwrapped
playback_controller = None
if fast_playback:
    if base.device.startswith("cuda"):
        from isaaclab_tasks.contrib.factory import factory_control
        from isaaclab.utils import math as torch_utils
        from playback_controller import PlaybackController
        playback_controller = PlaybackController(factory_control.get_pose_error, torch_utils)
        base._apply_action = lambda: playback_controller.apply(base)
    # Reward shaping and critic statistics are needed for PPO, not deployment.
    # Success is still checked against the official geometry at every action.
    playback_reward = torch.zeros(base.num_envs, device=base.device)
    def playback_rewards():
        base.prev_actions = base.actions.clone()
        return playback_reward
    base._get_rewards = playback_rewards
    print("PLAYBACK: saved actor, original contact physics, no training rewards.", flush=True)
(OUT/"workcell_origins.json").write_text(json.dumps(base.scene.env_origins.tolist(), indent=2))
if gui:
    import omni.usd
    from pxr import UsdUtils
    context = omni.usd.get_context()
    if context.get_stage() is not base.sim.stage:
        context.attach_stage_with_callback(UsdUtils.StageCache.Get().Insert(base.sim.stage).ToLongInt())
    for _ in range(8):
        app.update()
    import omni.ui
    for name in ("Stage","Property","Content","Console","Render Settings","IsaacLab","Isaac Lab"):
        window=omni.ui.Workspace.get_window(name)
        if window is not None:window.visible=False
    for _ in range(8):app.update()
    if a.render_quality == "fast":
        from omni.kit.viewport.utility import get_active_viewport_window
        viewport = get_active_viewport_window()
        if viewport is not None:
            viewport.viewport_api.set_hd_engine("rtx", "MinimalRendering")
            viewport.viewport_api.set_texture_resolution((960, 600))
            print("VIEWPORT RTX MODE:", viewport.viewport_api.render_mode, flush=True)
    print("LIVE RENDER:", a.render_quality, "every", render_every,
          "policy steps; physics dt", cfg.sim.dt, "decimation", cfg.decimation, flush=True)
    print("VISIBLE WORKCELL READY", flush=True)

def geometry():
    held, _ = factory_utils.get_held_base_pose(
        base.held_pos, base.held_quat, base.cfg_task.name,
        base.cfg_task.fixed_asset_cfg, base.num_envs, base.device)
    target, _ = factory_utils.get_target_held_base_pose(
        base.fixed_pos, base.fixed_quat, base.cfg_task.name,
        base.cfg_task.fixed_asset_cfg, base.num_envs, base.device)
    return torch.linalg.vector_norm(target[:,:2]-held[:,:2], dim=-1), held[:,2]-target[:,2]

agent_cfg = load_cfg_from_registry("IsaacContrib-Factory-GearMesh-Direct", "rl_games_cfg_entry_point")
params = agent_cfg["params"]
params["seed"] = a.seed
c = params["config"]
c.update(num_actors=a.num_envs, name="ParallelFranka", train_dir=str(OUT/"training"),
         full_experiment_name="visible_ppo")
wrapped = RlGamesVecEnvWrapper.from_agent_cfg(env, agent_cfg)
register_rl_games_env(wrapped)
stable = torch.zeros(a.num_envs, device=base.device, dtype=torch.long)
seen = torch.zeros(a.num_envs, device=base.device, dtype=torch.bool)
episode_rewards = torch.zeros(a.num_envs, device=base.device)
episode_records = []
success_geometry = {}
last_episode_success = {}
rollout_rewards = []
steps = 0
original_step = wrapped.step
def monitored_step(actions):
    global steps
    if playback_controller is not None:
        playback_controller.check()
    if not torch.isfinite(actions).all():
        raise RuntimeError("Policy produced non-finite actions.")
    current = base._get_curr_successes(base.cfg_task.success_threshold)
    stable.copy_(torch.where(current, stable+1, torch.zeros_like(stable)))
    newly_stable = (stable == 8).nonzero(as_tuple=False).flatten().tolist()
    if newly_stable:
        center, height = geometry()
        for idx in newly_stable:
            success_geometry[idx] = dict(center_error_mm=float(center[idx])*1000,
                                         height_above_target_mm=float(height[idx])*1000)
    seen.logical_or_(stable >= 8)
    result = original_step(actions)
    obs, reward, dones, info = result
    steps += 1
    episode_rewards.add_(reward.flatten())
    if not fast_playback:
        rollout_rewards.append(float(reward.mean()))
    for idx in dones.nonzero(as_tuple=False).flatten().tolist():
        rec = dict(environment=idx+1, success=bool(seen[idx]),
                   policy_step=steps, reward=float(episode_rewards[idx]))
        rec.update(success_geometry.get(idx, {}))
        last_episode_success[idx] = rec.copy()
        episode_records.append(rec)
        success_geometry.pop(idx,None)
        stable[idx] = 0
        seen[idx] = False
        episode_rewards[idx] = 0
    if steps % 4 == 0:
        center, height = geometry()
        rows = [dict(environment=i+1, center_error_mm=float(center[i])*1000,
                     height_above_target_mm=float(height[i])*1000,
                     reward=None if fast_playback else float(reward.flatten()[i]), stable_steps=int(stable[i]),
                     reached_stable_insertion=bool(seen[i])) for i in range(a.num_envs)]
        publish(policy_steps=steps, transitions=steps*a.num_envs, environments=rows,
                last_reward=None if fast_playback else float(reward.mean()), total_successes=sum(r["success"] for r in episode_records),
                total_trials=len(episode_records))
    return result
wrapped.step = monitored_step
runner = Runner(IsaacAlgoObserver())
baseline = ROOT/"checkpoints/nvidia_factory_gear_pretrained.pth"
checkpoint = Path(a.checkpoint) if a.checkpoint else baseline
randomization = dict(
    fixture_position_half_range_m=[.05,.05,.05], fixture_yaw_deg=[0,15],
    hand_position_half_range_m=[.02,.02,.01], hand_yaw_half_range_rad=.785,
    gear_in_gripper_half_range_m=[.003,0,.003],
    when="At episode reset; no teleporting during a policy-controlled insertion.",
    grasp="The reset supplies the held gear. Loose-part picking is a separate task.",
)
(OUT/"randomization_spec.json").write_text(json.dumps(randomization, indent=2))

if a.mode == "train":
    c.update(max_epochs=a.iterations, horizon_length=a.rollout, seq_length=32,
             minibatch_size=128, save_frequency=max(1,a.iterations//3), save_best_after=0)
    c["central_value_config"]["minibatch_size"] = 128
    params["load_checkpoint"] = False
    runner.load(agent_cfg)
    runner.reset()
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    if not a.from_scratch:
        agent.restore(str(checkpoint))
    agent.epoch_num, agent.frame = 0, 0
    before = {k:v.detach().cpu().clone() for k,v in agent.model.state_dict().items()}
    original_epoch = agent.train_epoch
    updates = []
    def measured_epoch():
        rollout_rewards.clear()
        publish(phase="Collecting experience", epoch=agent.epoch_num)
        old_mu = agent.model.a2c_network.mu.weight.detach().clone()
        result = original_epoch()
        def mean(values):
            return float(torch.stack([v.detach().float() for v in values]).mean())
        update = dict(epoch=agent.epoch_num, transitions=agent.frame+agent.curr_frames,
                      mean_step_reward=sum(rollout_rewards)/len(rollout_rewards),
                      actor_loss=mean(result[4]), critic_loss=mean(result[5]),
                      entropy=mean(result[7]), kl=mean(result[8]), learning_rate=float(result[9]),
                      mean_abs_action_head_weight_change=float(
                          (agent.model.a2c_network.mu.weight.detach()-old_mu).abs().mean()),
                      completed_trials=len(episode_records),
                      stable_successes=sum(r["success"] for r in episode_records),
                      wall_seconds=float(result[3]))
        updates.append(update)
        (OUT/"ppo_updates.json").write_text(json.dumps(updates, indent=2))
        publish(phase="PPO updated shared policy", epoch=agent.epoch_num, updates=updates)
        print("REAL PPO UPDATE", json.dumps(update), flush=True)
        return result
    agent.train_epoch = measured_epoch
    start = time.monotonic()
    agent.train()
    checkpoint_output = OUT/"checkpoints" if a.output_dir is not None else ROOT/"checkpoints"
    checkpoint_output.mkdir(exist_ok=True)
    trained = checkpoint_output/"franka_parallel_trained.pth"
    agent.save(str(trained.with_suffix("")))
    changed = [k for k,v in agent.model.state_dict().items() if not torch.equal(v.cpu(),before[k])]
    report = dict(task="IsaacContrib-Factory-GearMesh-Direct", new_ppo_epochs=agent.epoch_num,
                  transitions=agent.frame, parallel_environments=a.num_envs, rollout_steps=a.rollout,
                  starting_checkpoint=None if a.from_scratch else str(checkpoint),
                  starts_from_scratch=a.from_scratch, elapsed_seconds=time.monotonic()-start,
                  changed_model_tensors=changed, updates=updates, completed_training_trials=episode_records,
                  checkpoint=str(trained), checkpoint_sha256=hashlib.file_digest(trained.open("rb"),"sha256").hexdigest())
    assert len(changed)>0 and any(u["mean_abs_action_head_weight_change"]>0 for u in updates)
    (OUT/"training_report.json").write_text(json.dumps(report, indent=2))
    publish(phase="TRAINING COMPLETE", epoch=agent.epoch_num)
    print("TRAINING COMPLETE", json.dumps(report), flush=True)
else:
    runner.load(agent_cfg)
    player = runner.create_player()
    if not a.from_scratch:
        player.restore(str(checkpoint))
    player.reset()
    obs = wrapped.reset()
    obs = obs["obs"] if isinstance(obs,dict) else obs
    player.get_batch_size(obs,1)
    if player.is_rnn:
        player.init_rnn()
    print("GYM BATCH SHAPES", list(obs.shape), "actions", [a.num_envs,6], flush=True)
    print("RESET COMPLETE", flush=True)
    if a.mode in ("resets","snapshot"):
        for repeat in range(a.resets):
            if repeat:
                obs = wrapped.reset()
            publish(phase=f"Randomized reset {repeat+1} / {a.resets}")
            until = time.monotonic()+a.hold_seconds
            while app.is_running() and time.monotonic()<until:
                base.sim.render()
                app.update()
        from bake_scene import export_snapshot
        export_snapshot(base, OUT/"eight_randomized_workcells.usdc")
        (OUT/"snapshot_reset_samples.jsonl").write_text((OUT/"reset_samples.jsonl").read_text())
        publish(phase="Randomized workcells saved")
        print("RANDOMIZATION DEMO COMPLETE", flush=True)
    else:
        print("STARTING POLICY LOOP", flush=True)
        publish(phase="Running saved policy")
        counts = torch.zeros(a.num_envs,device=base.device,dtype=torch.long)
        records = []
        start = time.monotonic()
        trial_start, trial_first_step = start, steps
        while app.is_running() and (counts<a.episodes).any():
            with torch.inference_mode():
                action = player.get_action(player.obs_to_torch(obs),is_deterministic=True)
                obs, reward, dones, info = wrapped.step(action)
                obs = obs["obs"] if isinstance(obs,dict) else obs
                early = bool(seen.all())
                if early:
                    done_ids = torch.arange(a.num_envs,device=base.device)
                    for idx in done_ids.tolist():
                        if counts[idx]>=a.episodes:
                            continue
                        counts[idx] += 1
                        rec = dict(environment=idx+1,episode=int(counts[idx]),success=True,policy_step=steps,
                                   **success_geometry[idx])
                        rec.update(trial_wall_seconds=time.monotonic()-trial_start,
                                   trial_policy_steps=steps-trial_first_step)
                        records.append(rec)
                        print("ASSEMBLY RESULT",json.dumps(rec),flush=True)
                    seen[:] = False
                    stable[:] = 0
                    success_geometry.clear()
                    episode_rewards[:] = 0
                    if (counts<a.episodes).any():
                        obs = wrapped.reset()
                        obs = obs["obs"] if isinstance(obs,dict) else obs
                        trial_start, trial_first_step = time.monotonic(), steps
                else:
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    for idx in done_ids.tolist():
                        if counts[idx]>=a.episodes:
                            continue
                        counts[idx] += 1
                        previous = last_episode_success.get(idx, {})
                        ok = previous.get("success", False)
                        rec = dict(environment=idx+1,episode=int(counts[idx]),success=ok,policy_step=steps,completion="time_limit")
                        rec.update(trial_wall_seconds=time.monotonic()-trial_start,
                                   trial_policy_steps=steps-trial_first_step)
                        rec.update({k:previous[k] for k in ("center_error_mm","height_above_target_mm") if k in previous})
                        records.append(rec)
                        print("ASSEMBLY RESULT",json.dumps(rec),flush=True)
                    if len(done_ids):
                        trial_start, trial_first_step = time.monotonic(), steps
                if player.is_rnn:
                    for hidden in player.states:
                        hidden[:,done_ids,:] = 0
            if steps%25==0:
                print("POLICY PROGRESS",steps,flush=True)
        report = dict(checkpoint=str(checkpoint),seed=a.seed,parallel_environments=a.num_envs,
                      records=records,successes=sum(r["success"] for r in records),total=len(records),
                      elapsed_seconds=time.monotonic()-start,success_requirement="Official insertion geometry for 8 consecutive policy steps.")
        if a.mode == "deploy":
            report["playback_mode"] = a.playback_mode
            report["training_rewards_computed"] = not fast_playback
            report["physics"] = dict(
                timestep_seconds=cfg.sim.dt, steps_per_policy_action=cfg.decimation,
                solver_type=cfg.sim.physics.solver_type,
                max_position_iterations=cfg.sim.physics.max_position_iteration_count,
            )
            report["decorative_board_collisions"] = not fast_playback
        (OUT/(a.mode+"_report.json")).write_text(json.dumps(report,indent=2))
        publish(phase="DEPLOYMENT COMPLETE" if a.mode=="deploy" else "EVALUATION COMPLETE")
        print("EVALUATION COMPLETE",json.dumps(report),flush=True)
wrapped.close()
app.close()
