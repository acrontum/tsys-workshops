"""Capture cameras and ground-truth labels from eight actual randomized starts."""
import argparse
import json
import time
import faulthandler
import signal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"
p=argparse.ArgumentParser()
p.add_argument("--frames",type=int,default=16)
p.add_argument("--seed",type=int,default=2026)
p.add_argument("--resolution",type=int,default=640)
p.add_argument("--preview_delay",type=float,default=1.5)
a=p.parse_args()
faulthandler.register(signal.SIGUSR1,all_threads=True)
if a.frames<8 or a.resolution<64:
    p.error("Use at least eight frames and a resolution of at least 64.")
source=OUT/"eight_randomized_workcells.usdc"
if not source.is_file():
    raise SystemExit("Run ./workshop.sh randomize or snapshot first.")
dest=OUT/"synthetic_data"
dest.mkdir(exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"renderer":"RaytracedLighting"})
import numpy as np
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import Gf,Usd,UsdGeom
context=omni.usd.get_context()
# Keep all saved poses and material bindings, but render one detailed cell at a
# time. Eight simultaneously visible tables can exhaust RTX descriptors during
# segmentation even when sufficient VRAM remains available.
prepared=Usd.Stage.Open(str(source))
for idx in range(8):
    prepared.GetPrimAtPath(f"/World/envs/env_{idx}").SetActive(idx==0)
capture_stage=OUT/"single_visible_randomized_workcell.usdc"
prepared.GetRootLayer().Export(str(capture_stage))
if not context.open_stage(str(capture_stage)):
    raise RuntimeError("Cannot open the exported randomized workcells.")
while context.get_stage_loading_status()[2]>0:
    app.update()
for _ in range(8):app.update()
stage=context.get_stage()
origins=json.loads((OUT/"workcell_origins.json").read_text())
labels={"Robot":"franka_robot","FixedAsset":"gear_fixture","HeldAsset":"gear_medium",
        "SmallGearAsset":"gear_small","LargeGearAsset":"gear_large","Table":"workbench",
        "NIST_Taskboard":"nist_taskboard","NIST_ReferenceBase":"reference_fixture",
        "NIST_ReferenceSmall":"reference_gear_small","NIST_ReferenceMedium":"reference_gear_medium",
        "NIST_ReferenceLarge":"reference_gear_large"}
for idx in range(8):
    for name,label in labels.items():
        prim=stage.GetPrimAtPath(f"/World/envs/env_{idx}/{name}")
        if prim.IsValid():
            rep.functional.modify.semantics(prim,{"class":label},mode="replace")
rep.orchestrator.set_capture_on_play(False)
camera=UsdGeom.Camera.Define(stage,"/World/WorkshopDataCamera")
camera.GetFocalLengthAttr().Set(24.)
camera.GetHorizontalApertureAttr().Set(20.955)
camera.GetClippingRangeAttr().Set(Gf.Vec2f(.01,100.))
pose=camera.AddTransformOp()
product=rep.create.render_product(str(camera.GetPath()),(a.resolution,a.resolution))
backend=rep.backends.get("DiskBackend")
backend.initialize(output_dir=str(dest))
writer=rep.WriterRegistry.get("BasicWriter")
writer.initialize(backend=backend,rgb=True,instance_segmentation=True,
                  colorize_instance_segmentation=True,semantic_segmentation=True,
                  colorize_semantic_segmentation=True,distance_to_camera=True)
writer.attach(product)
rng=np.random.default_rng(a.seed)
manifest=dict(source_stage=str(source),seed=a.seed,frames=a.frames,
              resolution=[a.resolution,a.resolution],
              modalities=["rgb","semantic_segmentation","instance_segmentation","distance_to_camera"],
              purpose="Separate perception dataset; these images are not the state-based PPO policy input.",
              frozen_physics=True,randomized_workcells=8,
              render_strategy="One visible workcell per capture; all eight saved randomized poses are used.",
              captures=[])
print("SDG WRITER READY: real RGB, pixel labels and depth",flush=True)
for frame in range(a.frames):
    idx=frame%8
    for cell in range(8):
        stage.GetPrimAtPath(f"/World/envs/env_{cell}").SetActive(cell==idx)
    for name,label in labels.items():
        prim=stage.GetPrimAtPath(f"/World/envs/env_{idx}/{name}")
        if prim.IsValid():
            rep.functional.modify.semantics(prim,{"class":label},mode="replace")
    origin=np.array(origins[idx])
    eye=origin+np.array([.65,-1.15,.85])+rng.uniform([-.10,-.10,-.08],[.10,.10,.12])
    target=origin+np.array([.53,.04,.12])
    pose.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
    print("SDG FRAME SCHEDULE",frame,"environment",idx+1,flush=True)
    for _ in range(3):app.update()
    rep.orchestrator.step(rt_subframes=8,delta_time=0.)
    print("SDG RENDER COMPLETE",frame,flush=True)
    rep.orchestrator.wait_until_complete()
    for modality,extension in [("rgb","png"),("semantic_segmentation","png"),
                               ("instance_segmentation","png"),("distance_to_camera","npy")]:
        f=dest/f"{modality}_{frame:04d}.{extension}"
        if not f.is_file() or not f.stat().st_size:
            raise RuntimeError(f"Replicator did not create {f.name}")
    depth=np.load(dest/f"distance_to_camera_{frame:04d}.npy")
    finite=np.isfinite(depth)&(depth>0)
    preview=np.zeros(depth.shape,dtype=np.uint8)
    if finite.any():
        near,far=np.percentile(depth[finite],(2,98))
        far=max(far,near+.001)
        preview[finite]=((1-np.clip((depth[finite]-near)/(far-near),0,1))*255).astype(np.uint8)
    Image.fromarray(preview).save(dest/f"distance_to_camera_{frame:04d}_preview.png")
    row=dict(frame=frame,environment=idx+1,camera_world_position=eye.tolist(),
             look_at_world=target.tolist(),rgb=f"rgb_{frame:04d}.png",
             semantic=f"semantic_segmentation_{frame:04d}.png",
             depth_preview=f"distance_to_camera_{frame:04d}_preview.png")
    manifest["captures"].append(row)
    (dest/"workshop_manifest.json").write_text(json.dumps(manifest,indent=2))
    temp=OUT/"data_progress.tmp"
    temp.write_text(json.dumps(row|dict(completed=frame+1,total=a.frames),indent=2))
    temp.replace(OUT/"data_progress.json")
    print("SDG FRAME WRITTEN",json.dumps(row),flush=True)
    time.sleep(max(0,a.preview_delay))
writer.detach()
product.destroy()
print("SYNTHETIC DATA COMPLETE",a.frames,"frames; eight randomized starts; RGB + labels + depth.",flush=True)
app.close()
