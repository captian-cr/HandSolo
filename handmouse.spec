# -*- mode: python ; coding: utf-8 -*-
import os  # 只用于拼图标路径
#
# handmouse.spec —— PyInstaller 单文件（onefile）打包配置
#
# 【重要】此 spec 未经实机打包验证（未在真实环境跑过 pyinstaller）。
#         首次打包后请务必在**无 Python 的干净机器**上冒烟测试：
#         启动、摄像头采集、调试窗口、光标注入四条路径。
#
# 打包命令（在项目根目录执行）：
#     pyinstaller handmouse.spec
# 产物：dist/handmouse.exe（单文件，console 程序）。
#
# 注意事项 / 已知风险：
#   * 入口用 handmouse/__main__.py，它内部 `from handmouse.app import main`
#     并 `raise SystemExit(main())`，等价于 `python -m handmouse`。
#     （handmouse/app.py 只定义 main，作为入口脚本不会自动调用它。）
#   * datas 必须把 handmouse/models/hand_landmarker.task 放到
#     _MEIPASS/handmouse/models/ 下：vision.py 用
#     `Path(__file__).resolve().parent / "models" / "hand_landmarker.task"`
#     定位模型，冻结后 __file__ 位于 _MEIPASS/handmouse/，故相对目标必须
#     是 handmouse/models。缺了它启动时 pipeline 会报“缺少模型文件”。
#   * 未打包 config.json（有意为之）：config.py 的 _base_dir() 在冻结模式下
#     以 exe 自身所在目录为基准，因此 DEFAULT_CONFIG_PATH 就是
#     "<exe同级目录>/config.json"；首次运行若缺失，ConfigStore 会写入默认
#     配置，写在 exe 旁边不会丢失。需要把配置放到别处时用 `--config <路径>`。
#   * excludes 排掉的重依赖（PySide6/PyQt5/PyQt6/shiboken6/
#     IPython/pytest/setuptools）本项目运行路径根本没 import。其中
#     PySide6 虽在 ui extra 但 MVP 未实现。setuptools 被排有极小风险：
#     若将来某个依赖走 pkg_resources 会失败；目前闭包内无此用法。
#   * **不**排除 cv2/opencv-contrib-python、mediapipe、numpy、matplotlib：
#     前三个是运行必需；matplotlib 在 mediapipe 闭包里（见
#     requirements.lock），其视觉任务路径可能用到，排除收益小、风险高，
#     故保留。
#   * onefile 写法：单个 EXE(...) 直接吃 a.binaries/a.datas，**不写
#     COLLECT**（写了就变成 onedir 目录）。
#   * console=False：双击不弹黑窗口（与 pythonw 启动器一致）。代价是看不到 traceback，
#     所以日志必须落到 logs/，启动失败必须弹系统对话框 —— 都由
#     handmouse/__main__.py 负责（_ensure_output / _notify_no_console）。

block_cipher = None

datas = [
    # (源文件, 打包内的相对目标目录)；目标必须是 handmouse/models
    ("handmouse/models/hand_landmarker.task", "handmouse/models"),
]

# uiautomation 相关：
#   cursor_snap.py 在运行时懒加载 `from uiautomation import uiautomation`，
#   且被 try/except ImportError 包住，缺包时自动降级为纯坐标点击。
# 🔴 **仍然显式打包 uiautomation + comtypes**（2026-10-26 定）：
#   PyInstaller 的静态扫描其实**能**看到函数体内的 import 语句，所以只要构建
#   环境装了它，exe 就会带上（实测 PYZ 里 comtypes/uiautomation 齐全、磁吸可用）。
#   但那属于"碰巧抓到"：一旦构建环境少了这个包、或将来把 import 改成
#   importlib 动态加载，磁吸就会**静默消失** —— 打包不报错、启动不报错，
#   用户只会觉得"这版本残废"。缺依赖要崩在构建期，不能崩在用户手里。
#   实测 comtypes 在冻结版下能把 COM 类型库缓存写到
#   %TEMP%\comtypes_cache\HandSolo-<pid>，无需额外配置。
hiddenimports = ["uiautomation", "comtypes"]

excludes = [
    "PySide6",
    "PyQt5",
    "PyQt6",
    "shiboken6",
    # 🔴 **tkinter 不能排除**（2026-10-25 打包实测）：`handmouse/snap_ui.py`
    # 在模块顶层 `import tkinter`，而 app.py 会 import 到它 →
    # 排除后 exe 直接 `ModuleNotFoundError: No module named 'tkinter'` 起不来。
    # 这个排除项是打包配置写早了的遗留（那会儿还没有 snap_ui）。
    "IPython",
    "pytest",
    "setuptools",
]

a = Analysis(
    ["handmouse/__main__.py"],
    pathex=[SPECPATH],  # 项目根目录（spec 所在目录）作为模块搜索根
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

# onefile：EXE 同时包含 a.binaries 与 a.datas，且没有 COLLECT 段。
#: 应用图标：多尺寸 ico（Windows 资源管理器 / 任务栏 / 快捷方式都用它）。
#: 文件缺失时给 None —— PyInstaller 会用默认图标，不报错。
_ICON_PATH = os.path.join(SPECPATH, "assets", "hand-solo.ico")
_ICON = _ICON_PATH if os.path.exists(_ICON_PATH) else None

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HandSolo",
    icon=_ICON,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    # console=False：双击不弹黑窗口（否则用户第一反应是"这什么鬼"）。
    # 出错走 handmouse/__main__._notify_no_console 的系统弹窗 + logs/ 里的日志，
    # 与 pythonw 启动器同一套路径。
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
