"""诊断任意窗口的 UIA 控件树：看清"为什么这个软件吸不到"。

用法::

    python scripts/probe_app.py                 # 探测前台窗口
    python scripts/probe_app.py --title 豆包     # 探测标题含"豆包"的窗口
    python scripts/probe_app.py --grid 10       # 采样密度（默认 8，越大越慢）

输出三段：
1. 窗口基本信息（句柄、类名、矩形）——Electron 应用会显示 Chrome_WidgetWin_1；
2. 网格采样：每个点上的控件类型链 + 矩形，一眼看出是"真按钮"还是"一整块面板"；
3. 类型直方图：有没有 Button，全是 Custom/Pane 就是自绘 UI。

判断标准（磁吸能不能吸）：
* 出现 Button/MenuItem/TabItem/ListItem 等 → 能吸；
* 只有 Custom/Pane 但**矩形小**（宽<220 高<120）→ 新版靠尺寸兜底能吸；
* 整窗口只有一个大 Pane → 系统层面看不见，只能上图像识别，本项目不支持。
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections import Counter
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uiautomation import uiautomation as auto  # noqa: E402


def _visible_windows() -> list[tuple[int, str, str]]:
    user32 = ctypes.WinDLL("user32")
    found: list[tuple[int, str, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(handle, _param):
        if user32.IsWindowVisible(handle):
            length = user32.GetWindowTextLengthW(handle)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(handle, title, length + 1)
                cls = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(handle, cls, 128)
                found.append((int(handle), title.value, cls.value))
        return True

    user32.EnumWindows(callback, 0)
    return found


def _chain(control, depth: int = 6) -> str:
    parts: list[str] = []
    node = control
    for _ in range(depth):
        if node is None:
            break
        try:
            parts.append(node.ControlTypeName.replace("Control", ""))
        except Exception:
            break
        node = node.GetParentControl()
    return " → ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测窗口的 UIA 控件树")
    parser.add_argument("--title", default="", help="窗口标题包含的文字")
    parser.add_argument("--grid", type=int, default=8, help="横向/纵向采样点数")
    args = parser.parse_args(argv)

    windows = _visible_windows()
    if args.title:
        targets = [w for w in windows if args.title in w[1]]
    else:
        hwnd = int(ctypes.windll.user32.GetForegroundWindow())
        targets = [w for w in windows if w[0] == hwnd]
    if not targets:
        print(f"没找到标题含 {args.title!r} 的可见窗口。当前可见窗口：")
        for _handle, title, cls in windows[:20]:
            print(f"  {title[:40]!r:<44} class={cls!r}")
        return 1

    hwnd, title, cls = targets[0]
    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))

    # 前置检查：最小化或完全移出屏幕的窗口，采样全是 0×0 的空壳，
    # 得出来的"没暴露控件"是假结论。2026-10-20 用户就因此被误导过一次。
    if rect.left <= -30000 or rect.top <= -30000:
        print("窗口处于最小化状态（坐标 -32000 是 Windows 的哨兵值）。")
        print("请先把窗口正常显示在屏幕上，再重新运行。")
        return 2
    if rect.right <= 0 or rect.bottom <= 0 or rect.right <= rect.left:
        print("窗口没有有效尺寸（完全在屏幕外或不可见），测不到。")
        return 2

    print(f"窗口：{title!r}")
    print(f"  类名 {cls}   矩形 ({rect.left},{rect.top})-({rect.right},{rect.bottom})")
    if "Chrome_WidgetWin" in cls:
        print("  → Chromium/Electron 应用：控件树来自浏览器可访问性层，")
        print("    首次查询可能为空，第二次查询才会填充。")

    auto.SetGlobalSearchTimeout(0.6)
    width = max(1, rect.right - rect.left)
    height = max(1, rect.bottom - rect.top)
    types: Counter[str] = Counter()
    samples = max(2, args.grid)

    print(f"\n网格采样（{samples}×{samples}）：")
    seen: set[tuple[int, int, int, int]] = set()
    for row in range(samples):
        for col in range(samples):
            x = rect.left + int(width * (col + 0.5) / samples)
            y = rect.top + int(height * (row + 0.5) / samples)
            try:
                control = auto.ControlFromPoint(x, y)
            except Exception:
                continue
            if control is None:
                continue
            try:
                box = control.BoundingRectangle
                key = (box.left, box.top, box.right, box.bottom)
                name = str(control.Name or "")[:16]
                type_name = control.ControlTypeName
            except Exception:
                continue
            types[type_name] += 1
            if key in seen:
                continue
            seen.add(key)
            w = box.right - box.left
            h = box.bottom - box.top
            print(
                f"  ({x:>5},{y:>4}) {type_name:<20} {name!r:<18} "
                f"{w}×{h}    {_chain(control)}",
            )

    print("\n控件类型分布：")
    for name, count in types.most_common():
        print(f"  {name:<24} {count}")

    # 标题栏单独扫一遍：网格采样从 top + 高/16 起步，会跳过最顶上那一条，
    # 而 — □ × 正好在那里（2026-10-20 用户就关心这三个键）。
    print("\n标题栏横向扫描（顶部 +18px，从右边界往左）：")
    bar_y = rect.top + 18
    for offset in range(10, 210, 20):
        x = rect.right - offset
        try:
            control = auto.ControlFromPoint(x, bar_y)
        except Exception:
            continue
        if control is None:
            print(f"  x={x:>5} → 无控件")
            continue
        try:
            box = control.BoundingRectangle
            print(
                f"  x={x:>5} {control.ControlTypeName:<20} "
                f"{str(control.Name or '')[:14]!r:<16} "
                f"{box.right - box.left}×{box.bottom - box.top}",
            )
        except Exception:
            continue

    buttonish = sum(
        1
        for name in types
        if name in {
            "ButtonControl",
            "MenuItemControl",
            "TabItemControl",
            "ListItemControl",
            "TreeItemControl",
            "DataItemControl",
            "SplitButtonControl",
            "HyperlinkControl",
            "CheckBoxControl",
            "RadioButtonControl",
        }
    )
    print()
    if buttonish:
        print("结论：存在真正的可点控件 → 磁吸应当生效。")
    else:
        print("结论：没找到可点控件类型，只有自绘容器。")
        print("      若上面出现过 宽<220 高<120 的 Custom/Pane，新版按尺寸兜底能吸；")
        print("      若只有整窗口大的 Pane，UIA 层面无解，需要图像识别。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
