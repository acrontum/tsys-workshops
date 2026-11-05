#!/usr/bin/env python3
"""Open Isaac for a mouse-driven workshop; never execute the numbered cells."""
import builtins
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
builtins.WORKSHOP_ROOT = str(ROOT)
from isaacsim import SimulationApp
app = SimulationApp({'headless':False,'width':1920,'height':1080,'renderer':'RaytracedLighting'})
import carb.settings
import omni.kit.app
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport_window
settings = carb.settings.get_settings()
settings.set('/persistent/exts/omni.kit.window.script_editor/fontSize',18)
settings.set('/persistent/exts/omni.kit.window.script_editor/clearAfterExecute',False)
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate('omni.kit.window.script_editor',True)
for panel_name in ['Stage', 'Property']:
    ui.Workspace.show_window(panel_name, True)
script_extension = next(obj for obj in gc.get_objects() if type(obj).__name__=='ScriptEditorExtension' and hasattr(obj,'_window_list'))
omni.usd.get_context().new_stage()
for _ in range(20): app.update()
viewport = get_active_viewport_window()
keep = {viewport.title,'Main MenuBar','Toolbar','Stage','Property'}
for w in ui.Workspace.get_windows():
    if w.title not in keep: w.visible=False
viewport.position_x,viewport.position_y=40,35
viewport.width,viewport.height=1070,745
panel=ui.Window('WORKSHOP | Follow these actions',width=1895,height=245)
panel.position_x,panel.position_y=5,800
with panel.frame:
    with ui.VStack(spacing=8):
        title=ui.Label('PHYSICAL SORTING | Click, read the code, run, inspect',height=42,style={'font_size':29,'color':0xFFB8EB5A})
        caption=ui.Label('Open Window > Script Editor. The numbered files in cells/ build the scene and install the real controller.',
                         height=100,word_wrap=True,style={'font_size':25})
        note=ui.Label('Isaac Sim 6.2 | Reusable NVIDIA assets | Simulator-state sensing, no camera recognition',
                      height=60,word_wrap=True,style={'font_size':22,'color':0xFFD6CEB3})
control_mtime=None
editor=None
last_state=0.
try:
    while app.is_running():
        app.update()
        control=ROOT/'.tutorial_caption.json'
        if control.exists() and control.stat().st_mtime_ns != control_mtime:
            try:
                data=json.loads(control.read_text())
                title.text=data['title'];caption.text=data['caption'];note.text=data['note']
                control_mtime=control.stat().st_mtime_ns
            except (ValueError,KeyError): pass
        if time.monotonic()-last_state>.3:
            last_state=time.monotonic()
            editor=next((window for window in script_extension._window_list if window is not None and window.visible),None)
            handle=editor
            is_editor=editor is not None
            # Keep a readable layout, but opening/pasting/running/selection and
            # physics buttons are actual mouse and keyboard events from X11.
            if is_editor:
                for attr,value in [('position_x',1120),('position_y',35),('width',785),('height',745)]:
                    if getattr(handle,attr)!=value: setattr(handle,attr,value)
            for name,x,y,w,h in [('Stage',1120,35,785,350),('Property',1120,390,785,390)]:
                win=ui.Workspace.get_window(name)
                if win and win.visible is not None:
                    if win.visible != (not is_editor):
                        win.visible=not is_editor
                    if not is_editor:
                        for attr,value in [('position_x',x),('position_y',y),('width',w),('height',h)]:
                            if getattr(win,attr)!=value: setattr(win,attr,value)
            state={'windows': [{'title':w.title,'visible':w.visible,'x':w.position_x,'y':w.position_y,'width':w.width,'height':w.height} for w in ui.Workspace.get_windows()],
                   'selection':omni.usd.get_context().get_selection().get_selected_prim_paths(),
                   'editor_open':is_editor}
            if editor and editor._script_editor_widget:
                widget=editor._script_editor_widget
                state.update(editor_text=widget.current_tab.text if widget.current_tab else '',
                             editor_error=widget._has_run_error,editor_log=widget.get_last_log_text()[-4000:])
            sim=getattr(builtins,'_physical_workshop',None)
            if sim:
                state['simulation']=sim.snapshot()
            (ROOT/'.desktop_state.json').write_text(json.dumps(state,default=str))
            (ROOT/'.desktop_ready').touch()
except Exception:
    import traceback
    (ROOT/'desktop_error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close()
