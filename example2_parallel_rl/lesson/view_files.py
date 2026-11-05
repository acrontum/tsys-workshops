"""Read actual workshop files and generated images; never run selected source files."""
import json,tkinter as tk
from pathlib import Path
from PIL import Image,ImageTk
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
entries=[('Start here / README','README.md'),('Workshop commands','workshop.sh'),('NIST table and assets','scripts/workbench.py'),('RL task and training','scripts/parallel_workshop.py'),('RGB camera image','output/synthetic_data/rgb_0000.png'),('Semantic labels','output/synthetic_data/semantic_segmentation_0000.png'),('Depth preview','output/synthetic_data/distance_to_camera_0000_preview.png'),('Data manifest','output/synthetic_data/workshop_manifest.json'),('Training report','output/training_report.json'),('Evaluation results','output/evaluate_report.json'),('Deployment specification','output/deployment_spec.json'),('Robot trial results','output/deploy_report.json')]
r=tk.Tk();r.title('Parallel Workshop Files');r.geometry('1920x800+0+0');r.configure(bg='#101b28')
left=tk.Frame(r,bg='#182b3c',width=420);left.pack(side='left',fill='y');left.pack_propagate(False)
tk.Label(left,text='WORKSHOP FILES',font=('DejaVu Sans',23,'bold'),fg='#75deba',bg='#182b3c').pack(pady=18)
content=tk.Frame(r,bg='#101b28');content.pack(side='left',fill='both',expand=True)
title=tk.Label(content,text='',font=('DejaVu Sans',23,'bold'),fg='#75deba',bg='#101b28',anchor='w');title.pack(fill='x',padx=22,pady=14)
body=tk.Frame(content,bg='#101b28');body.pack(fill='both',expand=True,padx=20,pady=10)
buttons={};image=None

def show(path):
 global image
 for child in body.winfo_children():child.destroy()
 title.configure(text=path)
 p=ROOT/path
 if p.suffix=='.png' and p.exists():
  im=Image.open(p).convert('RGB');im.thumbnail((1120,660));image=ImageTk.PhotoImage(im);tk.Label(body,image=image,bg='#101b28').pack(anchor='w')
 else:
  t=tk.Text(body,font=('DejaVu Sans Mono',18),fg='#eef3f7',bg='#101b28',wrap='word',padx=8,pady=8,insertbackground='white')
  t.pack(fill='both',expand=True);t.insert('1.0',p.read_text() if p.exists() else 'This output is created when you run the corresponding workshop step.');t.configure(state='disabled')
 r.update_idletasks();write_state(path)
def write_state(path):
 rows={p:[b.winfo_rootx()+150,b.winfo_rooty()+22] for p,b in buttons.items()}
 (OUT/'viewer_state.json').write_text(json.dumps({'selected':path,'rows':rows},indent=2))
for label,path in entries:
 b=tk.Button(left,text=label,font=('DejaVu Sans',19),fg='#eef3f7',bg='#254054',activebackground='#426f83',anchor='w',command=lambda p=path:show(p),relief='flat',padx=18,pady=11);b.pack(fill='x',padx=12,pady=3);buttons[path]=b
show('README.md');r.mainloop()
