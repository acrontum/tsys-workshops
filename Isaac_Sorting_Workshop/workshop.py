"""Physical sorting runtime for the click-through workshop, Isaac Sim 6.2.

The spawner and routing context are shared in behavior with run_demo.py. The
numbered cells supply the detection monitor, destination function and control
callback explicitly. No animation trajectory or object pose-follow is used.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import asyncio
import builtins
import json
import time
import numpy as np
import omni.kit.app
import omni.usd
import omni.ui as ui
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate('isaacsim.robot_motion.examples', True)
import isaacsim.robot_motion.examples.manipulation.ur10_palletizing as nv
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux


def make_logic(stage, assets, materials, robot, count, destination_for):
    args = SimpleNamespace(count=count)
    class SortingSpawner(nv.BinSpawner):
        """Initialize each asset once; PhysX owns its pose after spawning."""
        def __init__(self, env_path, assets):
            super().__init__(env_path, assets)
            self.context = None
            self.travel = {}

        def step(self):
            if len(self.bins) >= args.count:
                return
            # One item at a time prevents an uncontrolled queue during pickup.
            if self.context is not None and len(self.bins) > len(self.context.stacked_bins):
                return
            i = len(self.bins)
            path = f'{self.env_path}/bins/bin_{i:02}'
            stage_utils.add_reference_to_stage(usd_path=self.assets.small_klt_usd, path=path)
            # Inverted KLT bins expose a flat suction surface; no flip maneuver is needed.
            body = RigidPrim(path, positions=[[0., 1.5, -.15]], orientations=[[0., 0., 1., 0.]],
                             reset_xform_op_properties=True)
            body.set_velocities(linear_velocities=[[0., 0., 0.]], angular_velocities=[[0., 0., 0.]])
            body.set_visibilities(True)
            prim = stage.GetPrimAtPath(path)
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not rigid_api or rigid_api.GetKinematicEnabledAttr().Get():
                raise RuntimeError(f'{path}: expected a dynamic rigid-body asset')
            lane = 'A' if i % 2 == 0 else 'B'
            prim.CreateAttribute('sorting:class', Sdf.ValueTypeNames.String).Set(lane)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[lane],
                bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            self.bins.append(body)
            self.on_conveyor = body
            self.travel[path] = {'initial_position': [0., 1.5, -.15], 'min_y': 1.5}
            print(f'PHYSICS_SPAWNED={path} CLASS={lane}', flush=True)

    class SortingContext(nv.BinStackingContext):
        def __init__(self, robot, spawner):
            super().__init__(robot, spawner, max_bins=args.count)
            self.counts = {'A': 0, 'B': 0}
            self.records = []
            self.grasps = set()
            self.monitor_target = None
            self.monitors.append(lambda ctx: ctx.assign_sort_target())

        def assign_sort_target(self):
            if self.active_bin is None:
                return
            path = str(self.active_bin.bin_obj.paths[0])
            lane = stage.GetPrimAtPath(path).GetAttribute('sorting:class').Get()
            slot = self.counts[lane]
            if slot >= 3:
                raise RuntimeError(f'Lane {lane} is full')
            # Two columns on the existing NVIDIA pallet: three positions per lane.
            target = destination_for(lane, slot)
            self.stack_coordinates[len(self.stacked_bins)] = target
            self.monitor_target = target
            if 'arrival_y' not in spawner.travel[path]:
                spawner.travel[path]['arrival_y'] = float(self.active_bin.bin_obj.get_world_poses()[0].numpy()[0,1])
            attached = robot.suction_gripper._iface.get_gripped_objects(robot.suction_gripper.surface_gripper_path)
            if any(path == p or p.startswith(path+'/') for p in attached):
                self.grasps.add(path)

        def apply(self, command, dt):
            # The stock example also zeros a bin's velocity when stopping the
            # belt. Omit that assist: friction must brake the dynamic bin.
            if command.conveyor_running is not None:
                self._set_conveyor_running(command.conveyor_running)
                command = replace(command, conveyor_running=None)
            super().apply(command, dt)

        def mark_active_bin_as_complete(self):
            path = str(self.active_bin.bin_obj.paths[0])
            lane = stage.GetPrimAtPath(path).GetAttribute('sorting:class').Get()
            position = np.asarray(self.active_bin.bin_obj.get_world_poses()[0].numpy()[0])
            if path not in self.grasps:
                raise RuntimeError(f'{path}: no real SurfaceGripper attachment was observed')
            if np.linalg.norm(position[:2]-self.monitor_target[:2]) > .12:
                raise RuntimeError(f'{path}: placement missed its assigned lane')
            self.records.append({'path': path, 'lane': lane, 'target': self.monitor_target.tolist(),
                                 'release_position': position.tolist(), 'grasp_confirmed': True})
            self.counts[lane] += 1
            super().mark_active_bin_as_complete()
            print(f'PHYSICS_SORTED={len(self.records)} LANES={self.counts}', flush=True)


    spawner = SortingSpawner('/World/PhysicalSorting', assets)
    context = SortingContext(robot, spawner)
    spawner.context = context
    context.reset()
    controller = nv.PalletizerController()
    controller.reset()
    return spawner, context, controller


def view_station():
    ViewportManager.set_camera_view(eye=[3.,3.,1.7], target=[.35,.2,-.35], camera='/OmniverseKit_Persp')


def new_stage():
    app_utils.stop()
    stage_utils.create_new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    UsdLux.DomeLight.Define(stage, '/World/BuildLight').CreateIntensityAttr(500)
    view_station()


def finish_environment(env, assets):
    """Add the remaining stock support/ground, collision filters and scenery."""
    ground = nv.GroundPlane(f'{env}/staticPlaneActor', sizes=50.,
                           translations=[(0.,0.,-1.1818999735824764)], templates=None)
    ground.set_visibilities(False)
    stage = omni.usd.get_context().get_stage()
    robot_group = UsdPhysics.CollisionGroup.Define(stage, f'{env}/CollisionGroup')
    robot_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path(f'{env}/ur10'))
    housing_group = UsdPhysics.CollisionGroup.Define(stage, f'{env}/CollisionGroup_01')
    housing_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path(f'{env}/sortbot_housing'))
    robot_group.CreateFilteredGroupsRel().AddTarget(housing_group.GetPrim().GetPath())
    stage_utils.add_reference_to_stage(usd_path=assets.background_usd, path='/World/Background')
    nv.XformPrim(['/World/Background'], positions=[10.,2.,-1.1818],
                 orientations=[.7071,0.,0.,.7071], reset_xform_op_properties=True)
    light = nv.DistantLight('/World/DistantLight', angles=np.deg2rad(1.),
                          orientations=[(.9238795,0.,.3826834,0.)])
    light.set_intensities(3000)
    stage.RemovePrim('/World/BuildLight')
    nv.create_planning_obstacles()
    view_station()


def disable_scenery_contacts(prim):
    if prim.IsInstance():
        prim.SetInstanceable(False)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    for child in prim.GetChildren():
        disable_scenery_contacts(child)


class Workshop:
    def __init__(self, robot, assets, count=6):
        self.robot, self.assets, self.count = robot, assets, count
        self.stage = omni.usd.get_context().get_stage()
        self.ready = self.success = False
        self.error = None
        self.completed_at = None
        self.last_state = None
        self.callback = None
        self.detector_installed = self.destination_installed = False
        self.output = Path(builtins.WORKSHOP_ROOT)/'result'
        self.output.mkdir(exist_ok=True)
        self.spawner = self.context = self.controller = None
        self.windows = []
        self.status_label = None
        self.started = time.monotonic()
        self.report = dict(success=False, mode='PhysX dynamics', requested_bins=count,
                           test_gripper=False, animation_time_samples=False,
                           perception='Simulator body poses and stored class IDs; no camera inference',
                           physics_engine='physx', asset_root=assets.assets_root_path,
                           reused_assets={k:v for k,v in vars(assets).items() if k.endswith('_usd')})
        builtins._physical_workshop = self

    async def initialize(self):
        try:
            while stage_utils.is_stage_loading():
                await omni.kit.app.get_app().next_update_async()
            disable_scenery_contacts(self.stage.GetPrimAtPath('/World/Background'))
            nv.configure_conveyor('/World/PhysicalSorting')
            materials = {}
            for lane, color in [('A',(.95,.32,.05)),('B',(.05,.65,.32))]:
                mat = UsdShade.Material.Define(self.stage, f'/World/SortingMaterials/{lane}')
                shader = UsdShade.Shader.Define(self.stage, str(mat.GetPath())+'/Surface')
                shader.CreateIdAttr('UsdPreviewSurface')
                shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color)
                shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.5)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
                materials[lane] = mat
            UsdGeom.SetStageMetersPerUnit(self.stage,1.)
            UsdGeom.SetStageUpAxis(self.stage,'Z')
            SimulationManager.set_physics_dt(1./60.)
            assert SimulationManager.get_active_physics_engine()=='physx'
            app_utils.play()
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            self.robot.initialize()
            self.robot.reset()
            self.spawner,self.context,self.controller = make_logic(self.stage,self.assets,materials,self.robot,
                        self.count,lambda lane,slot:self.destination_for(lane,slot))
            app_utils.pause()
            self.ready = True
            print('INITIALIZATION_OK: physics ready, controller paused')
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = str(e)

    def install_control_step(self, fn):
        assert self.ready and self.detector_installed and self.destination_installed
        if self.callback:
            SimulationManager.deregister_callback(self.callback)
        def guarded(dt,*args):
            if self.error or self.success:
                return
            try:
                fn(dt,*args)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.error = str(e)
                app_utils.pause()
        self.callback = SimulationManager.register_callback(guarded,event=SimulationEvent.PHYSICS_POST_STEP)
        print('CONTROL_CALLBACK_INSTALLED: read -> decide -> apply on each physics step')

    def observe(self, now):
        for body in self.spawner.bins:
            path = str(body.paths[0])
            y = float(body.get_world_poses()[0].numpy()[0,1])
            self.spawner.travel[path]['min_y'] = min(y,self.spawner.travel[path]['min_y'])
        if self.controller.state != self.last_state:
            self.last_state = self.controller.state
            print(f'CONTROLLER_STATE={self.last_state} t={now:.2f}')
        if self.status_label:
            self.status_label.text = f'{self.controller.state} | {len(self.context.records)}/{self.count} sorted\nA: {self.context.counts["A"]} | B: {self.context.counts["B"]}'
        if self.controller.state=='done':
            if self.completed_at is None:
                self.completed_at = now
                self.robot.suction_gripper.open()
            elif now-self.completed_at > 2.:
                self.save_and_validate(now)
        if now > 240 and not self.success:
            raise TimeoutError('Controller exceeded 240 simulated seconds')

    def save_and_validate(self, now):
        for record in self.context.records:
            body = next(b for b in self.spawner.bins if str(b.paths[0])==record['path'])
            record['settled_position'] = body.get_world_poses()[0].numpy()[0].tolist()
            record['settled_linear_speed'] = float(np.linalg.norm(body.get_velocities()[0].numpy()[0]))
        self.report.update(success=True, sorted_bins=len(self.context.records), lanes=self.context.counts,
                           placements=self.context.records, conveyor_travel=self.spawner.travel,
                           simulation_seconds=now,wall_seconds=time.monotonic()-self.started)
        self.stage.GetRootLayer().Export(str(self.output/'physical_sorting.usda'))
        (self.output/'report.json').write_text(json.dumps(self.report,indent=2)+'\n')
        from validate_report import validate
        try:
            validate(self.output/'report.json')
        except Exception:
            self.report['success'] = False
            (self.output/'report.json').write_text(json.dumps(self.report,indent=2)+'\n')
            raise
        self.success = True
        app_utils.pause()
        print('WORKSHOP_ACCEPTED: six physical attachments, correct lanes and settled bodies')

    def show_controls(self):
        assert self.callback
        w = ui.Window('SORTING CONTROLS',width=360,height=190)
        w.position_x,w.position_y = 700,50
        with w.frame:
            with ui.VStack(spacing=8):
                self.status_label = ui.Label('Ready | 0/6 sorted',height=45,style={'font_size':21})
                ui.Button('START / RESUME',height=42,clicked_fn=app_utils.play)
                ui.Button('PAUSE PHYSICS',height=42,clicked_fn=app_utils.pause)
        self.windows.append(w)
        print('CONTROLS_READY: click START / RESUME')

    def snapshot(self):
        return dict(ready=self.ready,success=self.success,error=self.error,
                    state=self.controller.state if self.controller else 'not initialized',
                    sorted=len(self.context.records) if self.context else 0,
                    spawned=len(self.spawner.bins) if self.spawner else 0,
                    sim_seconds=SimulationManager.get_simulation_time() if self.ready else 0.)
