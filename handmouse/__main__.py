"""支持 ``python -m handmouse``。

这里额外做两件事：

1. **单实例保护**（2026-10-23）：桌面双击和开机自启是两条入口，很容易同时
   开出两个实例抢摄像头——第二个会报"无法打开摄像头"，用户只会觉得"又坏了"。
   所以在进入主流程之前用 Windows 命名互斥量占个位：

   * 命名互斥量是内核对象，进程退出（含崩溃）时由系统自动释放，不会留下僵尸锁，
     比写 pid 文件可靠。
   * 已有实例时**不当错误**：提示一句、返回 0。双击的场景下就是"什么都没发生"，
     比弹一堆报错友好。

2. **无控制台时的输出兜底**（2026-10-23 轻量化部署）：启动器改用 ``pythonw.exe``
   之后 ``sys.stdout`` / ``sys.stderr`` 是 ``None``，任何 ``print`` 都会
   ``AttributeError`` 当场炸掉。这里在进入主流程**之前**把它们接到
   ``logs/handmouse-<日期>.log``，既不再崩，也让"崩了怎么查"这件事有个去处。
"""

from __future__ import annotations

import ctypes
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import IO

from handmouse import APP_NAME, APP_NAME_CN

#: 互斥量名字。只在本机、本用户语义下用，不需要 Global\\ 前缀。
#:
#: 🔴 改名（HandMouse → HandSolo）时**必须保留旧名字一起占**：过渡期如果旧版
#: 正在运行、新版没认出来，两个进程会同时抢摄像头并同时注入鼠标，光标当场乱飞。
_MUTEX_NAMES = (f"{APP_NAME}.SingleInstance.v1", "HandMouse.SingleInstance.v1")
_ERROR_ALREADY_EXISTS = 183


def _acquire_one(name: str) -> tuple[bool, int]:
    """占一个互斥量。返回 ``(是否抢到, 句柄)``；句柄 0 表示不适用或失败放行。"""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p,
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.CreateMutexW(None, 1, name)
    if not handle:
        # 拿不到互斥量（权限等罕见情况）就放行——保护机制本身不该把程序拦死。
        return True, 0
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return False, 0
    return True, int(handle)


def _acquire_single_instance() -> int | None:
    """尝试占位。成功返回互斥量句柄（0 表示不适用/降级放行），已存在返回 ``None``。"""

    if os.name != "nt":
        return 0
    keep = 0
    try:
        for name in _MUTEX_NAMES:
            ok, handle = _acquire_one(name)
            if not ok:
                return None
            keep = keep or handle
        return keep
    except Exception:
        return 0


def _ensure_output(log_dir: Path | None = None) -> Path | None:
    """把缺席的 stdout/stderr 接到日志文件。返回日志路径（本来有控制台则返回 ``None``）。

    ``log_dir`` 只给测试注入用；正常运行一律用项目根目录下的 ``logs/``。
    """

    if sys.stdout is not None and sys.stderr is not None:
        return None
    path: Path | None = None
    handle: IO[str]
    try:
        # 🔴 必须用 config.PROJECT_ROOT，不能自己拼 `__file__`：打包成单 exe 后
        # `__file__` 在 PyInstaller 的一次性解压目录里，日志写进去退出就没了
        # （2026-10-25 打包实测：窗口版双击 → 什么都没有，也查不到日志）。
        from handmouse.config import PROJECT_ROOT

        directory = log_dir or PROJECT_ROOT / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"handmouse-{time.strftime('%Y%m%d')}.log"
        handle = path.open("a", encoding="utf-8", buffering=1)
    except Exception:
        # 连日志都开不出来也不能让程序起不来：退到内存黑洞，至少不再 AttributeError。
        handle = io.StringIO()
        path = None
    if sys.stdout is None:
        sys.stdout = handle
    if sys.stderr is None:
        sys.stderr = handle
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {APP_NAME} 启动 ===")
    return path


def _notify_no_console(text: str) -> None:
    """无控制台时用系统弹窗把话说到——否则双击后"什么都没发生"就等于坏了。"""

    try:
        ctypes.WinDLL("user32").MessageBoxW(None, text, APP_NAME_CN, 0x40)
    except Exception:
        pass


def main() -> int:
    log_path = _ensure_output()

    if _acquire_single_instance() is None:
        message = (
            f"{APP_NAME_CN}（{APP_NAME}）已经在运行了。\n\n"
            "要接管它：对着摄像头比五指张开，保持约 1 秒；"
            "或按 Ctrl+Alt+G。\n要退出：Ctrl+Alt+Q。"
        )
        print(message)
        if log_path is not None:
            _notify_no_console(message)
        return 0

    if log_path is not None:
        print(f"无控制台启动（窗口版），输出已重定向到 {log_path}")

    try:
        from handmouse.app import main as app_main

        return app_main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        # pythonw 没有控制台：崩了如果不弹窗，用户看到的是"双击之后什么都没发生"，
        # 只会以为是我把软件改坏了。弹一句 + 指向日志，才算可诊断。
        traceback.print_exc()
        if log_path is not None:
            _notify_no_console(
                f"启动失败：{type(exc).__name__}: {exc}\n\n"
                f"完整日志：{log_path}\n"
                "想看实时输出，改用「HandSolo（诊断·看报错）.bat」。"
            )
        return 1


# 🔴 必须是 `if __name__ == "__main__"` 守卫，不能再写成模块级
# `raise SystemExit(main())`：那种写法让这个文件**无法被 import**——测试里
# 一 `import handmouse.__main__` 就会真的去启动摄像头和鼠标注入。
# `python -m handmouse` 本身会把本文件当 `__main__` 执行，守卫照常成立。
if __name__ == "__main__":
    raise SystemExit(main())
