"""Bake every Franka's measured body transforms for multi-environment Replicator."""
def export_snapshot(env,destination):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    stage=Usd.Stage.Open(env.sim.stage.Flatten())
    prims=list(stage.Traverse())
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    count=0
    for asset in env.scene.articulations.values():
        positions=asset.data.body_pos_w.torch.detach().cpu().numpy()
        quats=asset.data.body_quat_w.torch.detach().cpu().numpy()
        name=str(asset.cfg.prim_path).rsplit("/",1)[-1]
        for idx in range(env.num_envs):
            root=f"/World/envs/env_{idx}/{name}/"
            by_name={p.GetName():p for p in prims if str(p.GetPath()).startswith(root) and p.HasAPI(UsdPhysics.RigidBodyAPI)}
            for body_idx,body_name in enumerate(asset.body_names):
                prim=by_name.get(body_name)
                if prim is None:
                    continue
                x,y,z,w=map(float,quats[idx,body_idx])
                world=Gf.Matrix4d(1.)
                world.SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
                world.SetTranslateOnly(Gf.Vec3d(*map(float,positions[idx,body_idx])))
                local=world*cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
                transform=UsdGeom.Xformable(prim)
                transform.ClearXformOpOrder()
                transform.AddTransformOp().Set(local)
                cache.Clear()
                count+=1
    stage.GetRootLayer().Export(str(destination))
    assert count>=10*env.num_envs
    print("BAKED PHYSICS BODY POSES",count,flush=True)
