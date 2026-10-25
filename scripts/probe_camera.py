"""摄像头体检：换摄像头 / 画面不对时先跑这个。

只读工具——不改任何配置、不启动手势管线。它会如实回答三个问题：

1. **这只摄像头实际能吐哪些分辨率**（很多摄像头会"静默降级"：你要 1080p，
   它照样返回 True，但给你的还是 720p）；
2. **能不能协商 MJPG**（决定 USB 2.0 上高分辨率是 30fps 还是 10fps）；
3. **画面宽高比是多少**（4:3 与 16:9 会改变捏合比的标定尺度，程序里有自动
   补偿，但你需要知道它走的是哪条路）。

用法：

    python scripts/probe_camera.py                # 体检 0 号摄像头
    python scripts/probe_camera.py --index 1      # 体检 1 号摄像头
    python scripts/probe_camera.py --frames 90    # 测帧率时多抓几帧（更准）
    python scripts/probe_camera.py --backend any  # 只测默认后端

输出可以直接贴进 GitHub issue。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

#: 依次试探的档位。4:3 在前——本项目所有手势阈值都在 4:3 下标定。
CANDIDATES: tuple[tuple[int, int], ...] = (
    (320, 240),
    (640, 480),
    (800, 600),
    (960, 540),
    (1024, 768),
    (1280, 720),
    (1600, 1200),
    (1920, 1080),
)

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def _fourcc_text(value: float) -> str:
    code = int(value or 0)
    if code <= 0:
        return "（未协商）"
    text = "".join(chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    text = "".join(ch for ch in text if ch.isprintable()).strip()
    # MSMF 常常返回一串乱码/空串——那不是真 fourcc，别拿它当结论。
    return text if text.isalpha() and len(text) == 4 else "（读不到）"


def _sane(width: float, height: float) -> bool:
    return width >= 160 and height >= 120


def _open(index: int, backend: int) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        return None
    if os.name == "nt":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _measure_fps(capture: cv2.VideoCapture, frames: int) -> tuple[float, tuple[int, int] | None]:
    """抓 ``frames`` 帧量帧率；返回 (fps, 最后一帧尺寸)。"""

    for _ in range(5):  # 丢掉驱动缓冲里的旧帧
        capture.read()
    shape: tuple[int, int] | None = None
    count = 0
    started = time.perf_counter()
    deadline = started + 3.0  # 兜底：再慢也不测超过 3 秒
    while count < frames and time.perf_counter() < deadline:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        count += 1
        shape = (int(frame.shape[1]), int(frame.shape[0]))
    elapsed = max(1e-6, time.perf_counter() - started)
    return count / elapsed, shape


def probe(index: int, name: str, backend: int, frames: int) -> list[str]:
    lines: list[str] = []
    capture = _open(index, backend)
    if capture is None:
        lines.append(f"  ✗ 后端 {name}：打不开（可能被占用 / 无权限 / 不支持该后端）")
        return lines

    lines.append(f"  ✓ 后端 {name}：已打开")
    lines.append("")
    lines.append("    请求分辨率      驱动返回      状态")
    lines.append("    ------------    ------------  --------------------")

    delivered: list[tuple[int, int, int]] = []  # (w, h, 非原生标记)
    for width, height in CANDIDATES:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        got_w = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        got_h = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if not _sane(got_w, got_h):
            lines.append(
                f"    {width:>6}×{height:<5}    {got_w:>6}×{got_h:<5}    ✗ 驱动给的是垃圾值",
            )
            continue
        verdict = "原生" if (got_w, got_h) == (width, height) else f"静默改成 {got_w}×{got_h}"
        lines.append(f"    {width:>6}×{height:<5}    {got_w:>6}×{got_h:<5}    {verdict}")
        delivered.append((got_w, got_h, 0 if verdict == "原生" else 1))

    if delivered:
        native = [size for size in CANDIDATES if any(
            (got_w, got_h) == size for got_w, got_h, flag in delivered if flag == 0
        )]
        lines.append("")
        if native:
            lines.append(f"    原生档位：{', '.join(f'{w}×{h}' for w, h in native)}")
        else:
            lines.append("    原生档位：⚠️ 一个都没对上——这只摄像头会静默缩放，驱动不老实")

    # 帧率与真实画面：用 640×480 量（项目的默认档），拿不到就退到第一个能用的档。
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps, shape = _measure_fps(capture, frames)
    fourcc = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()

    lines.append("")
    if shape:
        aspect = shape[0] / shape[1]
        kind = "4:3" if abs(aspect - 4 / 3) < 0.05 else ("16:9" if abs(aspect - 16 / 9) < 0.05 else "非标准")
        lines.append(f"    实际画面：{shape[0]}×{shape[1]}  （宽高比 {aspect:.3f}，{kind}）")
        lines.append(f"    实测帧率：{fps:.1f} fps（{frames} 帧窗口）")
    else:
        lines.append("    实际画面：⚠️ 读不到帧，这只摄像头在这个后端下不可用")
    lines.append(f"    MJPG：{fourcc}")

    lines.append("")
    if shape and abs(shape[0] / shape[1] - 4 / 3) >= 0.05:
        lines.append(
            "    ⚠️ 画面不是 4:3：程序会自动把横向长度还原到 4:3 标定尺度"
            "（不用你改阈值），但捏合手感可能仍需微调 pinch_engage/pinch_release。",
        )
    if shape and fps < 20.0:
        lines.append(
            "    ⚠️ 帧率偏低：USB 2.0 摄像头在高分辨率下带宽吃满时就会掉到 ~10fps。"
            "把 camera_width/height 降到 640×480，或换 USB 3.0 接口。",
        )
    if fourcc in ("（未协商）", "（读不到）") or fourcc != "MJPG":
        lines.append(
            f"    ⚠️ 未协商到 MJPG（拿到 {fourcc}）：未压缩流会吃满 USB 2.0 带宽，"
            "高分辨率下会掉到 ~10fps。",
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="摄像头体检（只读，不改配置）")
    parser.add_argument("--index", type=int, default=0, help="摄像头序号，默认 0")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="dshow" if os.name == "nt" else "any",
        help="采集后端，默认 Windows 用 dshow；怀疑驱动有问题时试 msmf",
    )
    parser.add_argument("--frames", type=int, default=60, help="测帧率时抓多少帧，默认 60")
    args = parser.parse_args(argv)

    if os.name != "nt":
        print("提示：本程序只做 Windows 手势鼠标，其他系统下本脚本仅能测采集。")

    targets = [args.backend]
    if args.backend != "any" and os.name == "nt":
        # 顺手对比另一个后端：同一只摄像头在两个后端下的上限经常不一样。
        for other in ("dshow", "msmf"):
            if other != args.backend:
                targets.append(other)

    print(f"摄像头 #{args.index} 体检开始\n" + "=" * 62)
    any_ok = False
    for name in targets:
        print(f"\n[后端 {name}]")
        lines = probe(args.index, name, BACKENDS[name], max(10, args.frames))
        print("\n".join(lines))
        any_ok = any_ok or any("已打开" in line for line in lines)
    print("\n" + "=" * 62)
    if not any_ok:
        print("结论：这个序号上没有任何可用摄像头。试试 --index 1 / 2，")
        print("      或检查 设置 → 隐私 → 摄像头 是否允许桌面应用访问。")
        return 1
    print("结论：把上面的输出整段贴进 issue，就能判断该怎么配。")
    print("      推荐起点始终是 camera_width=640, camera_height=480, camera_fps=30。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
