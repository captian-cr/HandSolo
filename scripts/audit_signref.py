"""sign-ref 结构化日志审计：机械提取 H4 验收口径指标，输出 markdown 表格。

用法：
    python scripts/audit_signref.py --log logs/sign-ref-20261016-231852.log

输出为消费者是人（贴进对话/记录）的 markdown 表格，不是 JSON。

口径陷阱（改动前必读）：
1. 手背段「拒绝率」分母只计入非 SKIP 段。SKIP 段的帧若计入分母会得出
   "假 100%"（H4 三轮验收踩过：seg4 处于 gap3 carryover 时锁存未清零，
   其帧不能代表手背拒绝能力）。脚本按下列规则自动标 SKIP，不硬算：
     - 前置空洞存在但重入首帧非 state=lost（hand_lost 未触发 ≥200ms 无手
       判定，符号锁存未清零，下一段被上一段污染）
     - 段内姿势分类落在混合区（疑似两姿势融合成一个检测连续段——手未
       出画直接换姿势，无法归因到单一姿势）
     - 段帧数 < MIN_SEG_FRAMES（约 1s，姿势未保持）
2. 段界靠**时间空洞**自动检测，不读 stdout 的 [步骤] 时间窗（操作者可能
   滞后脚本数秒，231852 后半段慢约 4s），也不靠"无 palm 数据的帧"——
   无手时采集层不产包，日志整段静默（231852 间隙是 4.5s 零行）。空洞
   > GAP_MIN_S 才切段：亚秒级检测抖动不构成段界，而协议间隙为 4s。
   间隙的 hand_lost 判据 = 空洞后重入首帧 state==lost（引擎失联迁移）。
3. 姿势分类（掌/背）必须用**门无关**观测量：
   新格式下 palm_presented 是"进入判定的求值结果"，闸门一开即恒 False
   （见 gestures ① 语义变更），拿它分类会把已开闸的掌段误判成背段。
   故新格式按 nz_unit 符号对 expected_sign(hn, mirror) 的匹配率分类；
   旧格式（150336 及更早，早于语义变更）palm_presented 是原始几何值，
   仍按 palm_presented 比例分类。两者都是逐帧机械量，不读判定结果。
4. 观测量有三代格式：旧（cross/sin，150336 及更早）、中（nz_unit 无
   hn_score，语义变更后至 3c09481 之前）、新（nz_unit+hn_score）。
   中/旧格式能解析的字段照出，缺失字段标 N/A；三代观测量不可混读
   （见 gestures._log_sign_ref_facing）。中/旧格式均无 [环境] 指纹行。
5. 期望姿势按段序交替：seg1=掌、seg2=背、seg3=掌、seg4=背（新旧协议一致）。
   PASS/FAIL 由**期望**姿势驱动（手背段看 gate_active 帧数、掌心段看是否
   开闸）；实测分类只作交叉校验，与期望不符时照实标出由人判定——脚本
   只做机械提取，不替验收下最终结论。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
GEOM_RE = re.compile(r"geom ts=\d+ .*?gate_active=(True|False) .*?state=(\w+)")
PALM_RE = re.compile(r"palm_presented=(True|False) hn=(\S+) valid=(True|False)")
SIGN_NEW_RE = re.compile(
    r"\[sign-ref\] nz_unit=(-?[\d.]+) \|nz\|=([\d.]+)\(min=[\d.]+\) "
    r"hn=(\S*)(?: hn_score=([\d.]+))? frontal=[\d.]+ mirror=(?:True|False)"
)
SIGN_OLD_RE = re.compile(
    r"\[sign-ref\] cross=(-?[\d.]+) sin=([\d.]+)\(min=[\d.]+\) "
    r"hn=(\S+) frontal=[\d.]+ mirror=(?:True|False)"
)
HEADER_RE = re.compile(r"\[sign-ref运行头\] mirror=(True|False).*?(entry_palm_\w+)=([\d.]+)")
ENV_RE = re.compile(r"\[环境\] commit=(\S+) config_sha256=(\S+)")

#: 段帧数下限：低于此值视为姿势未保持（约 1s @30fps）。
MIN_SEG_FRAMES = 30
#: 姿势分类阈值：门无关判据匹配率 ≥HI 判掌、≤LO 判背、其间判混合（SKIP）。
MIXED_LO, MIXED_HI = 0.2, 0.8
#: 段界时间空洞下限（秒）。协议间隙 4s；亚秒级检测抖动不构成段界。
GAP_MIN_S = 1.5


@dataclass
class Frame:
    t: datetime
    gate_active: bool | None = None
    state: str = ""
    palm_presented: bool | None = None
    hn: str = ""
    valid: bool = False
    nz_unit: float | None = None
    abs_nz: float | None = None
    hn_score: float | None = None
    cross: float | None = None
    sin: float | None = None


@dataclass
class SegInfo:
    idx: int
    frames: list[Frame]
    n: int = 0
    dur: float = 0.0
    like_ratio: float = 0.0
    false_n: int = 0
    gate_on: int = 0
    reject: float = 0.0
    latency: float | None = None
    first: Frame | None = None
    cls: str = ""
    expected: str = ""
    skip: bool = False
    verdict: str = ""
    # 前置空洞指标（seg1 无前置空洞，保持默认值）
    gap_hole: float | None = None
    gap_lost: int = 0
    gap_handlost: bool = False


def read_text(path: str) -> str:
    """BOM sniff：PowerShell Tee-Object 落盘为 UTF-16LE，logging 落盘为 utf-8。"""

    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def parse(text: str) -> tuple[dict, list[Frame]]:
    header: dict = {}
    frames: list[Frame] = []
    for line in text.splitlines():
        m = TS_RE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
            microsecond=int(m.group(2)) * 1000
        )
        body = line[m.end():]
        if hm := HEADER_RE.search(body):
            header["mirror"] = hm.group(1) == "True"
            header["threshold"] = f"{hm.group(2)}={hm.group(3)}"
            continue
        if em := ENV_RE.search(body):
            commit, plus, _ = em.group(1).partition("+dirty")
            header["commit"] = commit
            header["dirty"] = bool(plus)
            header["config_sha256"] = em.group(2)
            continue
        if gm := GEOM_RE.search(body):
            frames.append(
                Frame(t=t, gate_active=gm.group(1) == "True", state=gm.group(2))
            )
            continue
        if not frames:
            continue
        f = frames[-1]
        if pm := PALM_RE.search(body):
            f.palm_presented = pm.group(1) == "True"
            f.hn = pm.group(2)
            f.valid = pm.group(3) == "True"
            continue
        if sn := SIGN_NEW_RE.search(body):
            f.nz_unit = float(sn.group(1))
            f.abs_nz = float(sn.group(2))
            if sn.group(4) is not None:
                f.hn_score = float(sn.group(4))
                header["hn_score"] = True
            header["format"] = "new"
            continue
        if so := SIGN_OLD_RE.search(body):
            f.cross = float(so.group(1))
            f.sin = float(so.group(2))
            header["format"] = "old"
    return header, frames


def segment(frames: list[Frame], header: dict) -> list[SegInfo]:
    """按时间空洞切检测连续段（无手时日志整段静默，见 docstring 口径 2）。"""

    if not frames:
        return []
    spells: list[list[Frame]] = [[frames[0]]]
    for prev, f in zip(frames, frames[1:]):
        if (f.t - prev.t).total_seconds() > GAP_MIN_S:
            spells.append([])
        spells[-1].append(f)
    segs: list[SegInfo] = []
    for n, spell in enumerate(spells):
        prev_last_t = spells[n - 1][-1].t if n > 0 else None
        segs.append(_analyze(n + 1, spell, prev_last_t, header))
    return segs


def _palm_like_ratio(frames: list[Frame], fmt: str, mirror: bool) -> tuple[float, int]:
    """门无关姿势判据匹配率（见 docstring 口径 3）。返回 (匹配率, 判据帧数)。"""

    if fmt == "old":
        judged = [f for f in frames if f.palm_presented is not None]
        if not judged:
            return 0.0, 0
        return sum(1 for f in judged if f.palm_presented) / len(judged), len(judged)
    judged = [f for f in frames if f.valid and f.nz_unit is not None and f.hn in ("Left", "Right")]
    if not judged:
        return 0.0, 0
    like = 0
    for f in judged:
        expected = 1.0 if f.hn == "Right" else -1.0
        if mirror:
            expected = -expected
        if (f.nz_unit > 0) == (expected > 0):
            like += 1
    return like / len(judged), len(judged)


def _analyze(
    idx: int, frames: list[Frame], prev_last_t: datetime | None, header: dict
) -> SegInfo:
    info = SegInfo(idx=idx, frames=frames)
    info.n = len(frames)
    info.dur = (frames[-1].t - frames[0].t).total_seconds()
    info.false_n = sum(1 for f in frames if f.palm_presented is False)
    info.reject = info.false_n / info.n
    info.gate_on = sum(1 for f in frames if f.gate_active)
    opened = next((f for f in frames if f.gate_active), None)
    info.latency = (opened.t - frames[0].t).total_seconds() if opened else None
    info.first = next(
        (f for f in frames if f.valid and (f.nz_unit is not None or f.cross is not None)),
        frames[0],
    )
    fmt = header.get("format", "")
    ratio, judged_n = _palm_like_ratio(frames, fmt, header.get("mirror", False))
    info.like_ratio = ratio
    if judged_n == 0:
        info.cls = "?"
    elif ratio >= MIXED_HI:
        info.cls = "掌"
    elif ratio <= MIXED_LO:
        info.cls = "背"
    else:
        info.cls = "混合"
    info.expected = "掌" if idx % 2 == 1 else "背"

    if prev_last_t is not None:
        info.gap_hole = (frames[0].t - prev_last_t).total_seconds()
        lost = 0
        for f in frames:
            if f.state != "lost":
                break
            lost += 1
        info.gap_lost = lost
        info.gap_handlost = lost > 0

    skips: list[str] = []
    if prev_last_t is not None and not info.gap_handlost:
        skips.append("前置空洞后 hand_lost 未触发（锁存未清零）")
    if info.cls == "混合":
        skips.append("姿势混合段，无法归因")
    if info.cls == "?":
        skips.append("无有效判据帧")
    if info.n < MIN_SEG_FRAMES:
        skips.append(f"帧数 {info.n} < {MIN_SEG_FRAMES}")
    info.skip = bool(skips)
    if skips:
        info.verdict = "SKIP: " + "；".join(skips)
    elif info.expected == "背":
        info.verdict = (
            "PASS" if info.gate_on == 0 else f"FAIL: 手背段 gate_active {info.gate_on} 帧"
        )
    else:
        info.verdict = "PASS" if info.latency is not None else "FAIL: 掌段未开闸"
    return info


def _first_basis(f: Frame, fmt: str) -> str:
    if fmt == "new":
        if f.nz_unit is None:
            return "N/A"
        score = f"{f.hn}@{f.hn_score:.3f}" if f.hn_score is not None else f.hn
        return f"nz={f.nz_unit:+.3f} \\|nz\\|={f.abs_nz:.2f} {score}"
    if f.cross is None:
        return "N/A"
    return f"cross={f.cross:+.4f} sin={f.sin:.3f} {f.hn}"


def render(header: dict, segs: list[SegInfo]) -> str:
    fmt = header.get("format", "")
    fmt_label = {"old": "旧(cross/sin)"}.get(fmt, "")
    if fmt == "new":
        fmt_label = "新(nz_unit/hn_score)" if header.get("hn_score") else "新(nz_unit)"
    if not fmt_label:
        fmt_label = "N/A"
    dirty = header.get("dirty")
    mirror = header.get("mirror")
    out = [
        "# sign-ref 审计",
        "",
        "## 运行头",
        "| commit | dirty | config_sha256 | mirror | 门槛 | 观测量 |",
        "|---|---|---|---|---|---|",
        "| {} | {} | {} | {} | {} | {} |".format(
            header.get("commit", "N/A"),
            "N/A" if dirty is None else dirty,
            header.get("config_sha256", "N/A"),
            "N/A" if mirror is None else mirror,
            header.get("threshold", "N/A"),
            fmt_label,
        ),
        "",
        "## 间隙（按时间空洞自动检测，hand_lost = 重入首帧 state=lost）",
        "| gap | lost 帧 | 空洞时长(s) | hand_lost 触发 | 后续段 |",
        "|---|---|---|---|---|",
    ]
    for s in segs:
        if s.gap_hole is None:
            continue
        out.append(
            f"| {s.idx - 1} | {s.gap_lost} | {s.gap_hole:.2f}"
            f" | {'✓' if s.gap_handlost else '✗'} | seg{s.idx} |"
        )

    out += [
        "",
        "## 分段",
        "| 段 | 期望 | 实测 | 帧数 | 时长(s) | 首帧判据 | gate_active 帧 | 拒绝率 | 开闸延迟(s) | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in segs:
        latency = "—" if s.latency is None else f"{s.latency:.2f}"
        out.append(
            f"| seg{s.idx} | {s.expected} | {s.cls} | {s.n} | {s.dur:.2f}"
            f" | {_first_basis(s.first, fmt)} | {s.gate_on} | {s.reject * 100:.1f}%"
            f" | {latency} | {s.verdict} |"
        )

    back = [s for s in segs if s.expected == "背" and not s.skip]
    palm = [s for s in segs if s.expected == "掌" and not s.skip]
    skips = [s for s in segs if s.skip]
    out += ["", "## 汇总（分母仅含非 SKIP 段）"]
    if back:
        gate_sum = sum(s.gate_on for s in back)
        false_sum = sum(s.false_n for s in back)
        total = sum(s.n for s in back)
        hard = "✓" if gate_sum == 0 else "✗"
        out.append(
            f"- 手背段 {len(back)} 段：gate_active 合计 {gate_sum} 帧（硬条件 {hard}）；"
            f"加权拒绝率 {false_sum / total * 100:.1f}%"
        )
    else:
        out.append("- 手背段：无非 SKIP 段——拒绝率分母为空，不给数值（防“假 100%”）")
    if palm:
        opened = sum(1 for s in palm if s.latency is not None)
        lat = " / ".join(f"{s.latency:.2f}s" for s in palm if s.latency is not None)
        out.append(
            f"- 掌心段 {len(palm)} 段：{opened}/{len(palm)} 开闸" + (f"，延迟 {lat}" if lat else "")
        )
    else:
        out.append("- 掌心段：无非 SKIP 段")
    out.append("- SKIP 段：" + ("、".join(f"seg{s.idx}" for s in skips) if skips else "无"))
    for s in segs:
        if not s.skip and s.cls != "?" and s.cls != s.expected:
            out.append(f"- ⚠ seg{s.idx} 实测分类（{s.cls}）与期望（{s.expected}）不符，请人工核对操作")
    if len(segs) != 4:
        out.append(f"- ⚠ 检测连续段数 {len(segs)} ≠ 协议期望 4，请人工核对间隙执行")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="sign-ref 结构化日志审计（口径陷阱见模块 docstring）"
    )
    ap.add_argument("--log", required=True, metavar="PATH", help="结构化 sign-ref 日志路径")
    args = ap.parse_args(argv)
    header, frames = parse(read_text(args.log))
    if not frames:
        print(
            "未解析到任何 geom 帧——确认传入的是结构化 sign-ref 日志（含 geom 行）。",
            file=sys.stderr,
        )
        return 2
    print(render(header, segment(frames, header)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
