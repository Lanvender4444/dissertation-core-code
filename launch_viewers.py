"""Small launcher for the existing, unchanged multi-camera viewers.

Double-click start_viewers.cmd, or run this script without arguments for a menu.
Only the Python standard library is imported; viewer dependencies stay in the
original ultralytics environment. No shell commands, fitting or model changes.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VIEWS = {
    'three': ('三相机：Cam2 / Cam3 / Cam4 ＋ 融合结果（无BAND）',
              'final_three_cam_no_band.py', (),
              '四画面：三个相机各自的检测结果＋三目融合。保留原评分、E1和当前帧／累计统计。'),
    'five': ('五相机：自然全景 ＋ BEV雷达图 ＋ 五路原图',
             'final_five_cam.py', ('--panorama', 'natural'),
             '原冻结五目结果的自然全景、BEV与五路原图。缓存重放，并统计本次访问帧。'),
    'bev': ('空间视角：3D环视 ＋ 纹理BEV（双栏）',
            'final_spatial_cam.py', (),
            '小屏分页界面：1总览、2五路原图、3对象信息；F全屏，V切自然全景。'),
    'bev-radar': ('BEV鸟瞰：独立大画面雷达图',
                  'final_immersive_cam.py', ('--page', 'bev'),
                  '直接进入大画面BEV，不挤在环绕图旁边；M回主视角、C看原图、F全屏。位置仍是原算法近似估计。'),
    'surround': ('主视角：沉浸环绕（全景／BEV／原图切页）',
                 'final_immersive_cam.py', (),
                 '鼠标拖动转头、滚轮缩放；M环绕、P等比例球面、C五路原图、G鸟瞰、F全屏。沿用原五目结果。'),
    'three-band': ('历史三相机：Cam2 / Cam3 / Cam4 ＋ 融合（BAND）',
                   'final_three_cam.py', (),
                   '以前的BAND＋深度＋E1四画面入口，单独保留，不与无BAND混为一种配置。'),
    'cam2': ('单独查看 Cam2 原始检测', 'final_cam2.py', (), '历史Cam2 YOLO对照，保留原自车排除。'),
    'cam3': ('单独查看 Cam3 原始检测', 'final_cam3.py', (), '历史Cam3 YOLO对照，保留原自车排除。'),
    'cam4': ('单独查看 Cam4 原始检测', 'final_cam4.py', (), '历史Cam4 YOLO对照，保留原自车排除。'),
}
THREE_VIEWS = {'three', 'three-band', 'cam2', 'cam3', 'cam4'}
VIEW_ALIASES = {'spatial': 'bev'}  # Preserve the original --view bev command.
QUICK_VIEWS = (('主视角', 'surround'), ('空间视角', 'bev'), ('BEV鸟瞰', 'bev-radar'))


def runner():
    uv = shutil.which('uv')
    if uv:
        return [uv, 'run', '--offline', '--no-sync', '--project', str(ROOT/'ultralytics'),
                'python', '-u', '-X', 'utf8']
    venv = ROOT/'ultralytics'/('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
    if venv.is_file():
        return [str(venv), '-u', '-X', 'utf8']
    raise RuntimeError('找不到uv或原项目虚拟环境。请在原项目环境运行，不会自动安装或升级依赖。')


def command(view, mode='auto', start='', fps=3., cache=False, *, headless=False,
            limit=0, save=False, out=None):
    view = VIEW_ALIASES.get(view, view)
    if view not in VIEWS or mode not in ('auto', 'step'):
        raise ValueError('请选择已有视角和播放模式。')
    if not math.isfinite(float(fps)) or not 0 < float(fps) <= 60:
        raise ValueError('播放帧率须在0到60之间。')
    stem = str(start).strip()
    if stem and (not stem.isascii() or not stem.isdigit() or len(stem) > 8):
        raise ValueError('起始帧填0到8位数字，例如30；留空表示从头开始。')
    if not isinstance(limit, int) or limit < 0:
        raise ValueError('limit须为非负整数。')
    if headless and mode != 'auto':
        raise ValueError('无窗口测试须使用auto模式。')
    if cache and view not in THREE_VIEWS:
        raise ValueError('五目入口已经是冻结缓存重放，不使用额外--cache。')
    entry = ROOT/'ultralytics/decided'/VIEWS[view][1]
    if not entry.is_file():
        raise FileNotFoundError(f'入口不存在：{entry}')
    cmd = runner()+[str(entry), *VIEWS[view][2], '--mode', mode, '--fps', str(float(fps)),
                    '--max-width', '1280', '--max-height', '720']
    if stem: cmd += ['--start', stem.zfill(8)]
    if cache: cmd += ['--cache']
    if headless: cmd += ['--headless']
    if limit: cmd += ['--limit', str(limit)]
    if save: cmd += ['--save']
    if out is not None: cmd += ['--out', str(Path(out).resolve())]
    return cmd


def start_viewer(cmd):
    # Inherit the launcher's console so statistics and failures stay readable
    # after a viewer exits. The OpenCV viewer itself creates its own window.
    # Popen receives an argument list, never shell=True or a string-built shell.
    env = os.environ.copy(); env['PYTHONUTF8'] = '1'; env['PYTHONIOENCODING'] = 'utf-8'
    return subprocess.Popen(cmd, cwd=ROOT, env=env)


class Launcher:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.root = root; self.child = None
        root.title('多相机融合 · 视角启动器')
        width = min(660, root.winfo_screenwidth()-60)
        height = min(590, root.winfo_screenheight()-90)
        root.geometry(f'{max(520,width)}x{max(450,height)}')
        root.minsize(520, 450)
        style = ttk.Style(root); style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 17, 'bold'))
        style.configure('Launch.TButton', font=('Microsoft YaHei UI', 12, 'bold'), padding=10)
        body = ttk.Frame(root, padding=20); body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1); body.rowconfigure(9, weight=1)
        ttk.Label(body, text='选择画面，一键启动', style='Title.TLabel').grid(row=0,column=0,sticky='w')
        ttk.Label(body, text='复用原来的Python入口，不改检测、融合或统计。').grid(row=1,column=0,sticky='w',pady=(4,18))
        self.keys = list(VIEWS)
        picker = ttk.Frame(body); picker.grid(row=2,column=0,sticky='ew')
        picker.columnconfigure(0,weight=1)
        self.combo = ttk.Combobox(picker, state='readonly', values=[v[0] for v in VIEWS.values()], height=len(VIEWS))
        self.combo.current(self.keys.index('surround')); self.combo.grid(row=0,column=0,sticky='ew')
        shortcuts = ttk.Frame(picker); shortcuts.grid(row=1,column=0,sticky='ew',pady=(8,0))
        self.quick_buttons = {}
        for column,(label,view) in enumerate(QUICK_VIEWS):
            shortcuts.columnconfigure(column,weight=1)
            button = ttk.Button(shortcuts,text=label,command=lambda key=view:self.choose(key))
            button.grid(row=0,column=column,sticky='ew',padx=(0,6) if column<2 else 0)
            self.quick_buttons[view] = button
        self.help = tk.StringVar()
        self.help_label = ttk.Label(body,textvariable=self.help,wraplength=590,justify='left')
        self.help_label.grid(row=3,column=0,sticky='ew',pady=(10,16))
        settings = ttk.Frame(body); settings.grid(row=4,column=0,sticky='ew')
        self.mode = tk.StringVar(value='auto'); self.start = tk.StringVar(); self.fps = tk.StringVar(value='3')
        ttk.Radiobutton(settings,text='自动播放',variable=self.mode,value='auto').grid(row=0,column=0,sticky='w')
        ttk.Radiobutton(settings,text='逐帧查看',variable=self.mode,value='step').grid(row=0,column=1,sticky='w',padx=14)
        ttk.Label(settings,text='起始帧').grid(row=1,column=0,sticky='w',pady=(12,0))
        ttk.Entry(settings,textvariable=self.start,width=12).grid(row=1,column=1,sticky='w',padx=14,pady=(12,0))
        ttk.Label(settings,text='播放FPS').grid(row=1,column=2,sticky='w',pady=(12,0))
        ttk.Entry(settings,textvariable=self.fps,width=6).grid(row=1,column=3,sticky='w',padx=8,pady=(12,0))
        self.cache = tk.BooleanVar(value=False)
        self.cache_button = ttk.Checkbutton(body,text='三目／单路复用已验证检测缓存（缺失会报错，不自动降级）',variable=self.cache)
        self.cache_button.grid(row=5,column=0,sticky='w',pady=(14,8))
        self.button = ttk.Button(body,text='启动所选视角',style='Launch.TButton',command=self.launch)
        self.button.grid(row=6,column=0,sticky='ew',pady=(8,8))
        self.status = tk.StringVar(value='就绪。默认环绕主视角；三目四画面对照仍可在菜单选择。')
        self.status_label = ttk.Label(body,textvariable=self.status,wraplength=590,justify='left')
        self.status_label.grid(row=7,column=0,sticky='ew')
        ttk.Separator(body).grid(row=8,column=0,sticky='ew',pady=12)
        self.notes = ttk.Label(body,text='查看器中：N下一帧 · B上一帧 · 空格播放/暂停 · Q退出\n'
            '五目/BEV/环视沿用原冻结输出；P/R仅累计本次访问帧。\n'
            '原图中的框与融合成员对应；单目漏检不会伪装成检测成功。\n'
            '关闭启动器不会强制结束已打开的查看器。',justify='left',wraplength=590)
        self.notes.grid(row=9,column=0,sticky='nw')
        self.combo.bind('<<ComboboxSelected>>',self.select)
        root.bind('<Configure>',self.resize)
        self.select(); root.after(600,self.poll)

    def selected(self):
        return self.keys[self.combo.current()]

    def choose(self, view):
        """Select a preset; preserve playback settings and explicit start button."""
        view = VIEW_ALIASES.get(view, view)
        self.combo.current(self.keys.index(view))
        self.select()

    def select(self, event=None):
        view = self.selected(); self.help.set(VIEWS[view][3])
        enabled = view in THREE_VIEWS
        self.cache_button.configure(state='normal' if enabled else 'disabled')
        if not enabled: self.cache.set(False)

    def resize(self, event):
        if event.widget is self.root:
            for label in (self.help_label,self.status_label,self.notes):
                label.configure(wraplength=max(420,event.width-45))

    def launch(self):
        from tkinter import messagebox
        if self.child is not None and self.child.poll() is None:
            messagebox.showinfo('查看器仍在运行','请先在当前查看器按Q退出，再启动其他视角。',parent=self.root)
            return
        try:
            cmd = command(self.selected(),self.mode.get(),self.start.get(),float(self.fps.get()),self.cache.get())
            self.child = start_viewer(cmd)
            self.status.set('已启动：'+VIEWS[self.selected()][0]+'。终端中保留原统计及错误信息。')
            self.button.configure(state='disabled')
        except (OSError,ValueError,RuntimeError) as error:
            messagebox.showerror('未能启动',str(error),parent=self.root)

    def poll(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.status.set('查看器已退出，可切换其他视角。' if code == 0 else
                                f'查看器退出码{code}，请查看对应终端错误；没有自动切换算法或模型。')
                self.child = None; self.button.configure(state='normal')
        self.root.after(600,self.poll)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=(*VIEWS, *VIEW_ALIASES))
    parser.add_argument('--mode',choices=('auto','step'),default='auto')
    parser.add_argument('--start',default=''); parser.add_argument('--fps',type=float,default=3.)
    parser.add_argument('--cache',action='store_true')
    parser.add_argument('--headless',action='store_true'); parser.add_argument('--save',action='store_true')
    parser.add_argument('--limit',type=int,default=0); parser.add_argument('--out',type=Path)
    parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--list',action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        for key,value in VIEWS.items(): print(f'{key}: {value[0]}')
        return 0
    if args.view:
        try:
            cmd = command(args.view,args.mode,args.start,args.fps,args.cache,headless=args.headless,
                          limit=args.limit,save=args.save,out=args.out)
        except (OSError,ValueError,RuntimeError) as error:
            parser.error(str(error))
        print(subprocess.list2cmdline(cmd),flush=True)
        return 0 if args.dry_run else subprocess.call(cmd,cwd=ROOT)
    if args.headless or args.dry_run or args.cache or args.limit or args.save or args.out or args.start:
        parser.error('命令行参数需要同时指定--view；无参数打开图形菜单。')
    import tkinter as tk
    root = tk.Tk(); Launcher(root); root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
