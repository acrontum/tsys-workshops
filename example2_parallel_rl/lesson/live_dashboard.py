"""Display measured Gym state and PPO statistics while Isaac runs."""
import json
import tkinter as tk
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"
r=tk.Tk()
r.title("Parallel RL Live Measurements")
r.geometry("560x800+1360+0")
r.bind("<Escape>", lambda event: r.destroy())
r.configure(bg="#101b28")
def label(text,size=20,color="#eef3f7",bold=False):
    w=tk.Label(r,text=text,font=("DejaVu Sans",size,"bold" if bold else "normal"),
               bg="#101b28",fg=color,anchor="w",justify="left",wraplength=520)
    w.pack(fill="x",padx=18,pady=3)
    return w
label("LIVE GYM + PPO",25,"#75deba",True)
phase=label("Waiting for Isaac",19,"#ffd27a",True)
epoch=label("Shared policy: eight robot copies",20,bold=True)
counts=label("Gym observations: [8, 19]\nGym actions: [8, 6]",18)
reward=label("New transitions: 0",18)
label("ENV     XY ERROR     HEIGHT",17,"#75deba",True)
rows=[label(f"{i+1:02d}         Waiting for reset",17) for i in range(8)]
loss=label("PPO statistics appear after an update.",17,"#ffd27a")
canvas=tk.Canvas(r,width=520,height=95,bg="#182b3c",highlightthickness=0)
canvas.pack(padx=18,pady=9)
note=label("Values come from the running simulator.\nA score is not an assembly success.",15,"#b7c9d9")
def update():
    try:
        state=json.loads((OUT/"live_state.json").read_text())
        phase.configure(text=state["phase"])
        if state["mode"]=="train":
            epoch.configure(text=f"Update {state['epoch']} / {state['epochs']}  |  Resets {state['resets']}")
        else:
            epoch.configure(text=f"{state['mode'].upper()}  |  Randomized resets {state['resets']}")
        description=(f"{state['num_envs']} environments share one policy"
                     if state["num_envs"]>1 else "One simulated Franka runs the saved policy")
        counts.configure(text=f"{description}\n{state['policy_steps']} batched Gym steps")
        last=state.get("last_reward")
        reward.configure(text=f"{state['transitions']:,} new transitions"+
                         (f"  |  Mean reward {last:.2f}" if last is not None else ""))
        envs=state.get("environments",[])
        starts=state.get("randomized_starts",[])
        for i,w in enumerate(rows):
            if i < len(envs):
                e=envs[i]
                w.configure(text=f"{i+1:02d}      {e['center_error_mm']:6.2f} mm     {e['height_above_target_mm']:6.2f} mm",
                            fg="#75deba" if e["reached_stable_insertion"] else "#eef3f7")
            elif i < len(starts):
                x,y,z=starts[i]["fixed_position"]
                w.configure(text=f"{i+1:02d}      target x {x:.3f} m, y {y:+.3f} m")
            else:
                w.configure(text=f"{i+1:02d}      —")
        updates=state.get("updates",[])
        if updates:
            u=updates[-1]
            loss.configure(text=f"PPO actor loss {u['actor_loss']:.4f}  |  KL {u['kl']:.5f}\n"
                           f"Measured weight change: {u['mean_abs_action_head_weight_change']:.6f}")
            values=[u["mean_step_reward"] for u in updates]
            lo,hi=min(values),max(values)
            if hi<=lo:hi=lo+1
            canvas.delete("all")
            canvas.create_text(12,12,text="Actual mean reward per rollout",fill="#eef3f7",anchor="w",font=("DejaVu Sans",12))
            points=[]
            for j,v in enumerate(values):
                x=18+j*480/max(1,state["epochs"]-1)
                y=80-(v-lo)/(hi-lo)*45
                points.extend([x,y])
                canvas.create_oval(x-3,y-3,x+3,y+3,fill="#75deba",outline="")
            if len(points)>=4:canvas.create_line(*points,fill="#75deba",width=2)
        r.lift()
    except (FileNotFoundError,json.JSONDecodeError,KeyError):
        pass
    r.after(400,update)
update()
r.mainloop()
