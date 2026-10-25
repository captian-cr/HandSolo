"""回放 usage 日志，量「原地停留（dwell）」到底能不能亮。

2026-10-23 用户反馈：「鼠标静止一秒钟也没有出现捏合的趋向」。
dwell 是磁吸窗口的唯一入口（见 ``gestures.snap_engaged``），
它不亮 → 磁吸永远不开 → 表现为"怎么都吸不到图标中央"。

回放结论（他自己 460 秒真实日志 / 14270 个 pointer 帧）：

* 旧判据 ``eps=0.005``（≈9.6px）**激活 0 次、占比 0.0%**。
  整段里最安静的 1 秒，从锚点的漂移还有 **24.8px**（p1=33.8px），
  而"静止"时逐帧抖动就有 2~4px —— 锚点式累计位移会一路累加，
  9.6px 几十毫秒就越界、计时永远清零，一秒阈值**物理上到不了**。
* 现判据（低通位置比半径激活 + 短窗/长窗位移解除）激活 2.3 次/分左右。

用法::

    python scripts/audit_dwell.py                    # 用最新日志
    python scripts/audit_dwell.py logs/usage-xxx.jsonl
"""

from __future__ import annotations

import copy
import glob
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from handmouse.config import AppConfig, ConfigStore  # noqa: E402
from handmouse.gestures import GestureEngine  # noqa: E402

#: 参与 dwell 的帧：只有 pointer 这一支会走 `_update_dwell`（见 gestures.py 指针分支）。
POINTER_GESTURES = {"pointer"}


def load_frames(path: Path) -> list[dict]:
    frames: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return frames


def make_engine(config: AppConfig) -> GestureEngine:
    """不走完整构造：只借 ``_update_dwell`` 这个方法本身，保证逻辑不漂移。"""

    engine = object.__new__(GestureEngine)
    engine.config = config
    engine._reset_dwell()
    return engine


def active_config() -> AppConfig:
    """优先用项目根目录的 ``config.json`` —— 那才是用户机器上真正生效的那份。

    只取默认值的 `AppConfig()` 会掩盖"配置文件里写死了旧阈值"这类问题：
    代码改成 0.030、JSON 里还是 0.005，回放看起来好了、他机器上依旧不亮。
    """

    path = PROJECT / "config.json"
    if path.exists():
        return ConfigStore(path).load()
    return AppConfig()


def runs(frames: list[dict]) -> list[list[dict]]:
    """把连续 pointer 帧切成段。段与段之间的空白会让计时自然清零。"""

    segments: list[list[dict]] = []
    current: list[dict] = []
    for frame in frames:
        if frame.get("g") in POINTER_GESTURES:
            current.append(frame)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def replay(segments: list[list[dict]], config: AppConfig) -> dict:
    engine = make_engine(config)
    activations = 0
    active_frames = 0
    longest_s = 0.0
    total_frames = 0

    def _run(segment: list[dict]) -> None:
        nonlocal activations, active_frames, longest_s, total_frames
        engine._reset_dwell()
        active_since: float | None = None
        for frame in segment:
            now_ns = int(float(frame["t"]) * 1_000_000_000)
            was_active = engine._dwell_active
            engine._update_dwell((float(frame["x"]), float(frame["y"])), now_ns)
            total_frames += 1
            if engine._dwell_active:
                active_frames += 1
                if not was_active:
                    activations += 1
                    active_since = float(frame["t"])
                elif active_since is not None:
                    longest_s = max(longest_s, float(frame["t"]) - active_since)
            else:
                active_since = None

    for segment in segments:
        _run(segment)
    return {
        "activations": activations,
        "active_ratio": active_frames / total_frames if total_frames else 0.0,
        "longest_s": longest_s,
        "frames": total_frames,
    }


def drift_profile(segments: list[list[dict]]) -> dict:
    """统计"一秒窗口内从锚点的最大位移"，用来说明旧阈值为什么不可达。"""

    samples: list[float] = []
    window_frames = 30  # 30fps 下约 1 秒
    for segment in segments:
        if len(segment) <= window_frames:
            continue
        for start in range(0, len(segment) - window_frames, 5):
            window = segment[start : start + window_frames]
            xs = [float(f["x"]) for f in window]
            ys = [float(f["y"]) for f in window]
            base_x, base_y = xs[0], ys[0]
            peak = max(
                ((x - base_x) ** 2 + (y - base_y) ** 2) ** 0.5
                for x, y in zip(xs, ys)
            )
            samples.append(peak)
    samples.sort()

    def pct(p: float) -> float:
        if not samples:
            return 0.0
        index = min(len(samples) - 1, int(p * len(samples)))
        return samples[index]

    return {
        "windows": len(samples),
        "min": samples[0] if samples else 0.0,
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
    }


def main() -> int:
    args = sys.argv[1:]
    if args:
        path = Path(args[0])
    else:
        found = sorted(glob.glob(str(PROJECT / "logs" / "usage-*.jsonl")), key=os.path.getmtime)
        if not found:
            print("没有找到 logs/usage-*.jsonl")
            return 1
        path = Path(found[-1])
    if not path.exists():
        print(f"文件不存在：{path}")
        return 1

    frames = load_frames(path)
    segments = runs(frames)
    pointer_frames = sum(len(s) for s in segments)
    duration = sum(s[-1]["t"] - s[0]["t"] for s in segments if len(s) > 1)
    print(
        f"日志 {path.name}：{len(frames)} 帧，其中 pointer {pointer_frames} 帧，"
        f"{len(segments)} 段，合计 {duration:.1f}s"
    )

    print("\n-- 1 秒窗口内从锚点的最大位移（旧判据的形状；归一化 → 1920 屏 px）--")
    profile = drift_profile(segments)
    if profile["windows"]:
        for key in ("min", "p10", "p25", "p50"):
            print(f"  {key:>3s}: {profile[key]:.4f}  ≈ {profile[key] * 1920:6.1f}px")
        print(
            f"  旧阈值 0.0050 (≈9.6px) 连最安静的窗口（{profile['min'] * 1920:.1f}px）都达不到"
        )
    else:
        print("  没有足够长的段")

    print("\n-- 现行判据（低通激活 + 短窗/长窗解除）--")
    base = active_config()
    result = replay(segments, base)
    per_min = result["activations"] / (duration / 60) if duration else 0.0
    print(
        f"  tau={base.pinch_pre_dwell_tau_s:.2f}s eps={base.pinch_pre_dwell_epsilon:.3f} "
        f"解除 {base.pinch_pre_dwell_release_ms}ms/{base.pinch_pre_dwell_release_epsilon:.3f} "
        f"+ {base.pinch_pre_dwell_drift_ms}ms/{base.pinch_pre_dwell_drift_epsilon:.3f}"
    )
    print(
        f"  激活 {result['activations']} 次  占 pointer 时间 "
        f"{result['active_ratio'] * 100:.1f}%  {per_min:.1f} 次/分  "
        f"最长连续 {result['longest_s']:.2f}s"
    )

    print("\n-- 灵敏度：只动 eps（其余取现行值）--")
    for eps in (0.015, 0.020, 0.030, 0.040):
        config = copy.deepcopy(base)
        config.pinch_pre_dwell_epsilon = eps
        single = replay(segments, config)
        rate = single["activations"] / (duration / 60) if duration else 0.0
        print(
            f"  eps={eps:.3f} (≈{eps * 1920:5.1f}px)  激活 {single['activations']:4d} 次  "
            f"占比 {single['active_ratio'] * 100:5.1f}%  "
            f"{rate:4.1f} 次/分  最长 {single['longest_s']:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
