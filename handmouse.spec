# -*- mode: python ; coding: utf-8 -*-
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
#   * excludes 排掉的重依赖（PySide6/PyQt5/PyQt6/shiboken6/tkinter/
#     IPython/pytest/setuptools）本项目运行路径根本没 import。其中
#     PySide6 虽在 ui extra 但 MVP 未实现。setuptools 被排有极小风险：
#     若将来某个依赖走 pkg_resources 会失败；目前闭包内无此用法。
#   * **不**排除 cv2/opencv-contrib-python、mediapipe、numpy、matplotlib：
#     前三个是运行必需；matplotlib 在 mediapipe 闭包里（见
#     requirements.lock），其视觉任务路径可能用到，排除收益小、风险高，
#     故保留。
#   * onefile 写法：单个 EXE(...) 直接吃 a.binaries/a.datas，**不写
#     COLLECT**（写了就变成 onedir 目录）。
#   * console=True：本程序是带 OpenCV 调试窗口的 CLI，保留控制台便于看日志。

block_cipher = None

datas = [
    # (源文件, 打包内的相对目标目录)；目标必须是 handmouse/models
    ("handmouse/models/hand_landmarker.task", "handmouse/models"),
]

# uiautomation 相关：
#   cursor_snap.py 在运行时懒加载 `from uiautomation import uiautomation`，
#   且被 try/except ImportError 包住，缺包时自动降级为纯坐标点击。
#   PyInstaller 静态分析抓不到这种运行期 import，因此默认不自动打包它。
#   这里**故意不加** hiddenimports("uiautomation")：
#     - 不加：构建 venv 有没有它都不影响 exe；目标机若没随 exe 带上，
#       运行期 import 失败会被 except 捕获 → 优雅降级，不会崩。
#     - 若确实要做 uia-enabled 单文件，再额外加
#       hiddenimports=["uiautomation", "comtypes"]（comtypes 是它的依赖，
#       且其 COM 类型库缓存机制在冻结环境下偶有问题，需实机验证）。
hiddenimports = []

excludes = [
    "PySide6",
    "PyQt5",
    "PyQt6",
    "shiboken6",
    "tkinter",
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
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="handmouse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
