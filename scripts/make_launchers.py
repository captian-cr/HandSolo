"""生成 .bat 启动器（GBK 编码）。

为什么单独写个脚本：cmd.exe 按 OEM 代码页读取 .bat，中文路径必须用 GBK 写，
用 UTF-8 会乱码 → cd 失败。用 Write 工具只能写 UTF-8，所以这里用 Python 转码。

生成五份（2026-10-23 改名 HandSolo + 去黑框）：

1. ``启动 HandSolo.bat``（项目目录）—— **主入口**，``pythonw.exe`` 启动，
   **没有黑窗口**，只有一个预览窗
2. ``HandSolo（后台无预览）.bat``（项目目录）—— 不带预览窗，最省 CPU
3. ``HandSolo（诊断·看报错）.bat``（项目目录）—— 用 ``python.exe`` 带控制台跑，
   崩了能直接看到 traceback。**pythonw 没有 stdout，出问题只能靠这一份**
4. 桌面一份 —— 双击即启动
5. 启动文件夹一份 —— 开机自启，默认暂停，比五指张开开始控制

🔴 **为什么不再用 ``python.exe`` + 黑窗口**：用户 2026-10-23 问「做成用终端打开的
形式好吗，GitHub 上下载的都没有这样的」。桌面工具的正确形态是双击即用、看不见
终端。``pythonw.exe`` 没有控制台，所以：
  * ``sys.stdout`` / ``sys.stderr`` 是 ``None`` → 由 ``handmouse/__main__.py``
    的 ``_ensure_output()`` 重定向到 ``logs/handmouse-<日期>.log``；
  * 启动失败时由 ``__main__`` 弹系统对话框告知，不会"双击后什么都没发生"。

🔴 **不要加 ``chcp 936``**：中文 Windows 本来就是 936，Python 打印中文走控制台
宽字符 API（WriteConsoleW），与代码页无关；改 chcp 反而会让重定向时按新代码页
编码，日志出现乱码。
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

#: 🔴 必须从**脚本位置**推导，不能写死。写死的那一版会让别人 clone 之后
#: 生成的启动器全指向作者自己的 `D:\...` 目录，双击必然失败（而且报错还被
#: pythonw 吞掉，看起来像"双击没反应"）。
PROJECT = Path(__file__).resolve().parent.parent
#: 用当前解释器，而不是猜一个路径：用户在虚拟环境里跑就指向那个 venv。
PYTHON = Path(sys.executable)
#: pythonw.exe 与 python.exe 同目录，没有控制台窗口。
PYTHONW = PYTHON.with_name("pythonw.exe")
if not PYTHONW.exists():
    # 非 Windows 或精简安装没有 pythonw 时退回 python.exe，仍然能跑。
    PYTHONW = PYTHON

MAIN_NAME = "启动 HandSolo.bat"
BACKGROUND_NAME = "HandSolo（后台无预览）.bat"
DIAG_NAME = "HandSolo（诊断·看报错）.bat"
SETUP_NAME = "HandSolo（首次设置）.bat"
DESKTOP_NAME = "HandSolo.bat"
STARTUP_NAME = "HandSolo（开机自启）.bat"

#: `start ""` 而不是 `start /b`：/b 会留在**父 cmd 的控制台里**，父进程一退出
#: 那个控制台被销毁，子进程可能跟着没了。`start ""` 拉起一个独立进程，而
#: pythonw.exe 本身没有控制台 → 全程看不见任何窗口。随后 `exit` 让这个 cmd
#: 立刻消失（只有约 100ms 的黑框闪现，这是 .bat 这个载体的固有代价）。
_LAUNCH = """@echo off
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
start "" "{pythonw}" -m handmouse{extra}
exit
"""

#: 桌面 / 启动文件夹那两份**不能**用 ``%~dp0``：它们的所在目录是桌面和启动文件夹，
#: 在那里 ``-m handmouse`` 根本导不进来，还会在桌面上造一个 ``logs/``。
#: 所以这里写死项目绝对路径。
_LAUNCH_REMOTE = """@echo off
cd /d "{project}"
start "" "{pythonw}" -m handmouse{extra}
exit
"""

#: 诊断入口：保留控制台，报错看得见。
#: 首次设置：**必须 python.exe**（要看得到"录了多少帧/写到哪"的反馈，
#: 而 pythonw 没有控制台）。它自己会开一个 OpenCV 窗口做标点与练习引导。
_SETUP = """@echo off
title HandSolo（首次设置）
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
echo.
echo [HandSolo] 首次设置：把程序调成你的手。
echo   1) 画面上会出现一个十字，把手指移过去、用最舒服的姿势捏合
echo   2) 随意动一动手指，记录你自己的习惯速度
echo   3) 每种手势带你做一遍
echo.
echo 全程不会动你的鼠标。随时按 ESC 退出。
echo.
"{python}" -m handmouse --onboarding
if errorlevel 1 (
    echo.
    echo [HandSolo] 设置没有完成，上面是原因。按任意键关闭。
    pause >nul
)
"""

_DIAG = """@echo off
title HandSolo（诊断模式）
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
echo [HandSolo] 诊断模式：这个窗口会一直显示运行日志，关掉窗口即退出。
echo 比五指张开（保持约 1 秒）开始控制；Ctrl+Alt+G 也可接管。
echo 退出：预览窗里按 Q，或 Ctrl+Alt+Q。
echo.
"{python}" -m handmouse
if errorlevel 1 (
    echo.
    echo [HandSolo] 启动失败，上面是错误原因。按任意键关闭。
    pause >nul
)
"""

#: 旧版留下的启动器名字，生成新文件时顺手清掉，免得桌面/项目目录里堆一堆。
STALE_PROJECT = (
    "启动手势鼠标.bat",
    "手势鼠标（后台无预览）.bat",
    "手势鼠标（带预览）.bat",
)
STALE_DESKTOP = ("手势鼠标.bat",)
STALE_STARTUP = ("手势鼠标（开机自启）.bat",)


def write_gbk(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.open(path, "w", encoding="gbk", newline="\r\n").write(text)
    print(f"写入 {path}")


def _drop(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        target = directory / name
        if target.exists():
            target.unlink()
            print(f"删除旧启动器 {target}")


def main() -> None:
    for required in (PYTHON, PYTHONW):
        if not required.exists():
            raise SystemExit(f"找不到 {required}")
    if not (PROJECT / "handmouse").is_dir():
        raise SystemExit(f"找不到包目录：{PROJECT / 'handmouse'}")

    desktop = Path(os.path.expandvars(r"%USERPROFILE%\Desktop"))
    startup = Path(
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup")
    )

    main_bat = _LAUNCH.format(pythonw=PYTHONW, extra="")
    write_gbk(PROJECT / MAIN_NAME, main_bat)
    write_gbk(
        PROJECT / BACKGROUND_NAME,
        _LAUNCH.format(pythonw=PYTHONW, extra=" --no-preview"),
    )
    write_gbk(
        PROJECT / DIAG_NAME,
        _DIAG.format(python=PYTHON),
    )
    write_gbk(
        PROJECT / SETUP_NAME,
        _SETUP.format(python=PYTHON),
    )
    remote = _LAUNCH_REMOTE.format(project=PROJECT, pythonw=PYTHONW, extra="")
    write_gbk(desktop / DESKTOP_NAME, remote)
    write_gbk(startup / STARTUP_NAME, remote)

    _drop(PROJECT, STALE_PROJECT)
    _drop(desktop, STALE_DESKTOP)
    _drop(startup, STALE_STARTUP)

    print("完成")


if __name__ == "__main__":
    main()
