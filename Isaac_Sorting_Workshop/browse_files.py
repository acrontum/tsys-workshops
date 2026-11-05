#!/usr/bin/env python3
"""Simple, real file viewer for the beginner workshop. It never runs the code."""
from pathlib import Path
import tkinter as tk
import json
from tkinter.scrolledtext import ScrolledText
ROOT=Path(__file__).resolve().parent
app=tk.Tk();app.title('Workshop Files - Read and copy the actual source');app.geometry('1920x800+0+0');app.configure(bg='#101b28')
tk.Label(app,text='1. Choose a file    2. Read its actual contents    3. Copy entire file    4. Paste into Isaac and click Run',
         font=('DejaVu Sans',19),fg='#e0e8f0',bg='#101b28').pack(fill='x',pady=10)
body=tk.Frame(app,bg='#101b28');body.pack(fill='both',expand=True)
left=tk.Frame(body,bg='#101b28',width=390);left.pack(side='left',fill='y');left.pack_propagate(False)
files=[ROOT/'README.txt']+sorted((ROOT/'cells').glob('*.py'))+[ROOT/'workshop.py',ROOT/'start_isaac.py',ROOT/'open_isaac.sh']
paths=tk.Listbox(left,font=('DejaVu Sans Mono',17),width=33,bg='#192b3b',fg='#f2f6fa',selectbackground='#275c52',activestyle='none')
paths.pack(fill='both',expand=True,padx=8,pady=8)
for path in files:paths.insert('end',('cells/' if path.parent.name=='cells' else '')+path.name)
right=tk.Frame(body,bg='#101b28');right.pack(side='left',fill='both',expand=True)
name=tk.Label(right,text='Choose a file on the left',font=('DejaVu Sans',20),fg='#75deba',bg='#101b28',anchor='w');name.pack(fill='x',pady=6)
text=ScrolledText(right,font=('DejaVu Sans Mono',17),bg='#101b28',fg='#e0e8f0',insertbackground='white',wrap='none',undo=False)
text.pack(fill='both',expand=True,padx=6)
current=None

def load(event=None):
 global current
 selection=paths.curselection()
 if not selection:return
 current=files[selection[0]]
 text.configure(state='normal');text.delete('1.0','end');text.insert('1.0',current.read_text());text.configure(state='disabled')
 name.configure(text=str(current.relative_to(ROOT)))

def copy():
 if current:
  app.clipboard_clear();app.clipboard_append(current.read_text());app.update()
  status.configure(text='Copied the complete file. Now paste into the Isaac Script Editor.')
paths.bind('<<ListboxSelect>>',load)
copy_button=tk.Button(right,text='COPY ENTIRE FILE',command=copy,font=('DejaVu Sans',18),bg='#75deba',fg='#101b28');copy_button.pack(anchor='w',pady=8,padx=6)
status=tk.Label(right,text='This viewer reads real files. It never executes Python.',font=('DejaVu Sans',15),fg='#becbd6',bg='#101b28');status.pack(fill='x',pady=5)
def telemetry():
 app.update_idletasks()
 rows={str(p.relative_to(ROOT)):[paths.winfo_rootx()+120,paths.winfo_rooty()+paths.bbox(i)[1]+paths.bbox(i)[3]//2] for i,p in enumerate(files) if paths.bbox(i)}
 data={'rows':rows,'copy':[copy_button.winfo_rootx()+90,copy_button.winfo_rooty()+18],'current':str(current.relative_to(ROOT)) if current else None}
 (ROOT/'.browser_state.json').write_text(json.dumps(data))
 app.after(250,telemetry)
paths.selection_set(0);load();app.after(500,telemetry);app.mainloop()
