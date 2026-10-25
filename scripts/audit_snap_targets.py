"""磁吸目标审查：把当前屏幕上**所有**可吸目标逐个量一遍，找出"吸不到中央"的原因。

为什么要这个脚本（2026-10-23 用户原话）：
> 「把能够磁吸到的每个小标都审查一遍，因为每次都会被吸到图标的不同位置，
>   很难被吸到小标的中央」

只读，不动鼠标、不改系统。做四件事：

1. **枚举**：遍历前台窗口的 UI Automation 树，收集所有符合磁吸白名单的控件
   （Button / TabItem / ListItem / TreeItem / DataItem / MenuItem / …），
   外加桌面图标（它们不在 UIA 树里，走 ``read_desktop_icons``）。
2. **量几何**：每个目标的矩形、宽高、面积、中心点、UIA 点击点，
   以及**点击点偏离中心多少像素**（``to_center`` 关掉时就是它在抖）。
3. **查冲突**：矩形重叠超过一半的、互相嵌套的、中心几乎重合的，
   这三类是"明明在 A 上却被 B 吸走"的直接原因。
4. **查抖动**：在目标矩形内撒 5×5 网格点，逐点跑真实的 ``_best_candidate``
   排序，看同一个图标内部会不会被吸到**不同**目标上。

用法::

    python scripts/audit_snap_targets.py                # 审查前台窗口 + 桌面
    python scripts/audit_snap_targets.py --radius 34    # 指定磁吸半径
    python scripts/audit_snap_targets.py --desktop-only # 只看桌面图标

输出：控制台摘要 + ``logs/snap-audit-<时间戳>.csv``（可拿 Excel 打开逐条看）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from handmouse.cursor_snap import (  # noqa: E402
    SnapCandidate,
    WindowsUiAutomationProbe,
    _best_candidate,
    _contains,
    _desktop_at,
)

#: 复用探测器的白名单，避免这里和产品代码各维护一份而漂移。
CONTROL_TYPES: set[int] = set(WindowsUiAutomationProbe._CONTROL_TYPES)

MAX_NODES = 4000
MAX_DEPTH = 10


@dataclass
class Target:
    name: str
    control_type: str
    left: int
    top: int
    right: int
    bottom: int
    depth: int = 0
    parent: str = ""
    click_x: int | None = None
    click_y: int | None = None
    source: str = "UIA"
    notes: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[int, int]:
        return (
            int(round((self.left + self.right) / 2)),
            int(round((self.top + self.bottom) / 2)),
        )

    @property
    def click_offset(self) -> float | None:
        if self.click_x is None or self.click_y is None:
            return None
        cx, cy = self.center
        return hypot(self.click_x - cx, self.click_y - cy)

    def as_candidate(self) -> SnapCandidate:
        cx, cy = self.center
        return SnapCandidate(
            x=self.click_x if self.click_x is not None else cx,
            y=self.click_y if self.click_y is not None else cy,
            left=self.left,
            top=self.top,
            right=self.right,
            bottom=self.bottom,
            name=self.name,
            control_type=self.control_type,
        )

    def label(self) -> str:
        return f"{self.control_type}「{self.name or '(无名)'}」"


# ---------------------------------------------------------------- 枚举


def walk_uia(auto: object, root: object) -> list[Target]:
    """深度优先遍历 UIA 树，收集白名单控件。"""

    found: list[Target] = []
    stack: list[tuple[object, int, str]] = [(root, 0, "")]
    visited = 0
    while stack and visited < MAX_NODES:
        control, depth, parent_name = stack.pop()
        visited += 1
        if depth > MAX_DEPTH:
            continue
        try:
            type_id = int(control.ControlType)
        except Exception:
            continue
        if type_id in CONTROL_TYPES:
            target = _read_control(control, depth, parent_name)
            if target is not None:
                found.append(target)
        try:
            children = control.GetChildren()
        except Exception:
            children = []
        try:
            child_name = str(control.Name or "")[:24]
        except Exception:
            child_name = parent_name
        for child in reversed(children or []):
            stack.append((child, depth + 1, child_name))
    return found


def _read_control(control: object, depth: int, parent: str) -> Target | None:
    try:
        if not bool(control.IsEnabled) or bool(control.IsOffscreen):
            return None
        rect = control.BoundingRectangle
        left, top = int(rect.left), int(rect.top)
        right, bottom = int(rect.right), int(rect.bottom)
        if right <= left or bottom <= top:
            return None
        try:
            name = str(control.Name or "")[:40]
        except Exception:
            name = ""
        try:
            type_name = str(control.ControlTypeName or "?")
        except Exception:
            type_name = "?"
        click_x = click_y = None
        try:
            cx, cy, ok = control.GetClickablePoint()
            if ok:
                click_x, click_y = int(cx), int(cy)
        except Exception:
            pass
        return Target(
            name=name,
            control_type=type_name,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            depth=depth,
            parent=parent,
            click_x=click_x,
            click_y=click_y,
        )
    except Exception:
        return None


def desktop_targets() -> list[Target]:
    """桌面图标：不在 UIA 树里，直接用列表项矩形。"""

    try:
        from handmouse.grid_desktop import read_desktop_icons
    except Exception:
        return []
    try:
        icons = read_desktop_icons()
    except Exception:
        return []
    return [
        Target(
            name=icon.name,
            control_type="DesktopIcon",
            left=icon.left,
            top=icon.top,
            right=icon.right,
            bottom=icon.bottom,
            source="desktop",
        )
        for icon in icons
    ]


# ---------------------------------------------------------------- 分析


def _reachable(target: Target) -> bool:
    """这个目标在当前屏幕状态下真的可能被吸到吗？

    🔴 生产代码里 ``DesktopIconProbe.find_target`` 会先过 ``_desktop_at(x, y)``：
    光标那个点上露出来的必须是桌面本身（没被别的窗口盖住），否则直接不返回候选。
    审查脚本**必须复刻这道闸**，否则会报出"桌面图标和窗口按钮重叠"这种
    根本不会发生的假冲突（2026-10-23 第一版就犯了这个错）。
    """

    if target.source != "desktop":
        return True
    return _desktop_at(*target.center)


def _overlap_ratio(a: Target, b: Target) -> float:
    """重叠面积 ÷ 较小者面积。1.0 = 小的那个被完全盖住。"""

    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    smaller = min(a.area, b.area) or 1
    return inter / smaller


def mark_conflicts(targets: list[Target]) -> list[str]:
    """给每个目标打上"和谁撞了"的标记，返回全局摘要行。"""

    lines: list[str] = []
    live = [t for t in targets if _reachable(t)]
    for t in targets:
        if t not in live:
            t.notes.append("被窗口挡住，当前状态吸不到")
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            ratio = _overlap_ratio(a, b)
            if ratio <= 0.5:
                continue
            nested = _contains(a.as_candidate(), b.as_candidate())
            reverse = _contains(b.as_candidate(), a.as_candidate())
            if nested or reverse:
                inner, outer = (b, a) if nested else (a, b)
                inner.notes.append(f"嵌在 {outer.label()} 里")
                outer.notes.append(f"内含 {inner.label()}")
                kind = "嵌套"
            else:
                a.notes.append(f"与 {b.label()} 重叠 {ratio:.0%}")
                b.notes.append(f"与 {a.label()} 重叠 {ratio:.0%}")
                kind = "重叠"
            cx1, cy1 = a.center
            cx2, cy2 = b.center
            lines.append(
                f"  [{kind}] {a.label()} {a.width}×{a.height}@({a.left},{a.top})"
                f"  ×  {b.label()} {b.width}×{b.height}@({b.left},{b.top})"
                f"  重叠{ratio:.0%}  中心相距 {hypot(cx1 - cx2, cy1 - cy2):.0f}px"
            )
    return lines


def flapping_report(targets: list[Target], step: int = 5) -> list[str]:
    """在一个目标内部撒网格，看会不会被吸到别的目标上。

    这正是用户说的"每次被吸到不同位置"：如果同一个图标内部不同点吸到的
    目标不一样，那落点当然每次都不同。
    """

    candidates = [t.as_candidate() for t in targets]
    desktop_ids = {
        id(c) for c, t in zip(candidates, targets) if t.source == "desktop"
    }
    lines: list[str] = []
    for index, target in enumerate(targets):
        if target.width < 6 or target.height < 6:
            continue
        winners: dict[str, int] = {}
        for gx in range(step):
            for gy in range(step):
                x = target.left + int((gx + 0.5) * target.width / step)
                y = target.top + int((gy + 0.5) * target.height / step)
                # 复刻生产闸：光标点上必须是桌面露出来，桌面图标才算候选。
                desktop_ok = _desktop_at(x, y)
                if target.source == "desktop" and not desktop_ok:
                    # 这一格已被窗口盖住，用户指到的其实是那个窗口，
                    # 不是这个图标。算进抖动就是冤案。
                    continue
                nearby = [
                    c
                    for c in candidates
                    if c.left - 2 <= x <= c.right + 2
                    and c.top - 2 <= y <= c.bottom + 2
                    and (desktop_ok or id(c) not in desktop_ids)
                ]
                if not nearby:
                    continue
                best = _best_candidate(x, y, nearby)
                key = f"{best.control_type}「{best.name or '(无名)'}」"
                winners[key] = winners.get(key, 0) + 1
        if len(winners) > 1:
            detail = "、".join(f"{k}×{v}" for k, v in sorted(winners.items(), key=lambda kv: -kv[1]))
            lines.append(
                f"  [抖动] {target.label()} 内部 {target.width}×{target.height}"
                f" 会吸到 {len(winners)} 个不同目标：{detail}"
            )
            target.notes.append(f"内部会吸到 {len(winners)} 个目标")
        elif winners:
            only = next(iter(winners))
            if only != f"{target.control_type}「{target.name or '(无名)'}」":
                lines.append(
                    f"  [被抢] {target.label()} 内部全部被 {only} 吸走"
                )
                target.notes.append(f"整块被 {only} 吸走")
    return lines


def write_csv(targets: list[Target], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "来源", "控件类型", "名称", "左", "上", "右", "下",
                "宽", "高", "中心X", "中心Y", "UIA点击X", "UIA点击Y",
                "点击点偏离中心px", "层级", "父节点", "问题",
            ],
        )
        for t in targets:
            cx, cy = t.center
            offset = t.click_offset
            writer.writerow(
                [
                    t.source, t.control_type, t.name, t.left, t.top, t.right, t.bottom,
                    t.width, t.height, cx, cy,
                    t.click_x if t.click_x is not None else "",
                    t.click_y if t.click_y is not None else "",
                    f"{offset:.1f}" if offset is not None else "",
                    t.depth, t.parent, "；".join(t.notes),
                ],
            )


# ---------------------------------------------------------------- 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="磁吸目标审查（只读）")
    parser.add_argument("--radius", type=int, default=34, help="磁吸半径像素，默认 34")
    parser.add_argument("--grid", type=int, default=5, help="抖动检测网格边长，默认 5×5")
    parser.add_argument("--desktop-only", action="store_true", help="只审查桌面图标")
    parser.add_argument("--no-desktop", action="store_true", help="不审查桌面图标")
    args = parser.parse_args()

    targets: list[Target] = []
    window_title = ""

    if not args.desktop_only:
        try:
            import uiautomation as auto

            control = auto.GetForegroundControl()
            window_title = str(getattr(control, "Name", "") or "")
            targets.extend(walk_uia(auto, control))
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 读取 UIA 树失败：{exc}")

    if not args.no_desktop:
        targets.extend(desktop_targets())

    if not targets:
        print("没有找到任何可吸目标。")
        print("提示：先让目标窗口露在最前面，再跑一次；UIA 读不到时请确认 handmouse[uia] 已安装。")
        return 1

    conflict_lines = mark_conflicts(targets)
    flap_lines = flapping_report(targets, step=max(3, args.grid))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = PROJECT / "logs" / f"snap-audit-{stamp}.csv"
    write_csv(targets, csv_path)

    # ---- 控制台摘要 ----
    print("=" * 78)
    print(f"磁吸目标审查  前台窗口「{window_title[:40]}」  半径 {args.radius}px")
    print(f"共 {len(targets)} 个目标（UIA {sum(1 for t in targets if t.source == 'UIA')}，"
          f"桌面 {sum(1 for t in targets if t.source == 'desktop')}）")
    print("=" * 78)

    print("\n-- 逐个目标 --")
    for t in targets:
        cx, cy = t.center
        offset = t.click_offset
        extra = f" UIA点击点偏离中心 {offset:.0f}px" if offset and offset >= 2 else ""
        print(
            f"  {t.label():<46} {t.width:>4}×{t.height:<4} "
            f"中心({cx},{cy}){extra}"
        )
        if t.notes:
            print(f"      ↳ {'；'.join(t.notes)}")

    print("\n-- 冲突 --")
    print("\n".join(conflict_lines) if conflict_lines else "  没有重叠/嵌套冲突。")

    print("\n-- 抖动（同一个目标内部会不会吸到别的目标）--")
    print("\n".join(flap_lines) if flap_lines else "  没有抖动，落点稳定。")

    print(f"\n明细已写入 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
