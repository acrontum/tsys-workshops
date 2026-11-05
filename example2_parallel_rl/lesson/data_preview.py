"""Display actual files immediately after the Replicator writer creates them."""
import json
import tkinter as tk
from pathlib import Path
from PIL import Image,ImageTk
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"
r=tk.Tk();r.title("Parallel Synthetic Data");r.geometry("1920x800+0+0")
r.bind("<Escape>", lambda event: r.destroy())
r.configure(bg="#101b28")
title=tk.Label(r,text="Waiting for the first real camera capture",font=("DejaVu Sans",29,"bold"),
               bg="#101b28",fg="#75deba");title.pack(pady=18)
panels=tk.Frame(r,bg="#101b28");panels.pack(fill="x",padx=20)
widgets=[]
placeholder=tk.PhotoImage(width=580,height=580)
for name in ("RGB: camera colour","Semantic labels: part classes","Depth preview: near is bright"):
    panel=tk.Frame(panels,bg="#182b3c");panel.pack(side="left",padx=10)
    tk.Label(panel,text=name,font=("DejaVu Sans",21,"bold"),bg="#182b3c",fg="white").pack(pady=12)
    widget=tk.Label(panel,bg="#182b3c",image=placeholder,width=580,height=580);widget.pack()
    widgets.append(widget)
note=tk.Label(r,text="The labels come from the known 3D scene. PPO uses simulator states, not these images.",
              font=("DejaVu Sans",22),bg="#101b28",fg="#ffd27a");note.pack(pady=16)
images=[];last=None
def refresh():
    global images,last
    try:
        row=json.loads((OUT/"data_progress.json").read_text())
        if row["frame"]!=last:
            images=[]
            for w,key in zip(widgets,("rgb","semantic","depth_preview")):
                im=Image.open(OUT/"synthetic_data"/row[key]).convert("RGB").resize((580,580))
                photo=ImageTk.PhotoImage(im);images.append(photo);w.configure(image=photo,width=580,height=580)
            title.configure(text=f"ACTUAL WRITER OUTPUT  |  Capture {row['completed']} / {row['total']}  |  Environment {row['environment']}")
            last=row["frame"]
    except (FileNotFoundError,json.JSONDecodeError):
        pass
    r.after(250,refresh)
refresh();r.mainloop()
