"""命令行入口。

为什么没有 GUI：MVP 阶段一切围绕"先把引擎跑通、先 safe-mode 验证"。配置走
``config.json`` + 少量 CLI flags，交互走 OpenCV 调试窗口和全局热键，不引入
PySide6，也就不会有打包体积和线程-事件循环耦合的问题。

用法：
    python -m handmouse                    # 启动，默认暂停，按 M 接管鼠标
    python -m handmouse --safe-mode        # 不注入任何输入，只跑状态机
    python -m handmouse --diagnostic-seconds 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from math import hypot
from pathlib import Path
from types import SimpleNamespace

from handmouse import __version__, debug_view, guided
from handmouse.config import PROJECT_ROOT, AppConfig, ConfigStore
from handmouse.hotkeys import HotkeyMonitor
from handmouse.launch import LaunchDispatcher
from handmouse.mouse import MouseController, NullMouseBackend, Win32MouseBackend
from handmouse.pipeline import GesturePipeline
from handmouse.screen_map import ScreenMap, dpi_state, system_dpi_scale

LOGGER = logging.getLogger(__name__)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handmouse",
        description="HandMouse 视觉手势鼠标（引擎层移植自 Vimoose，MIT）",
    )
    parser.add_argument("--config", metavar="PATH", help="配置文件路径，默认仓库根目录 config.json")
    parser.add_argument("--camera", type=int, help="摄像头序号，覆盖 config")
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="不注入任何系统输入，只跑状态机与调试窗口",
    )
    parser.add_argument(
        "--auto-control",
        action="store_true",
        help="启动即接管鼠标（默认先暂停，按 M 或 Ctrl+Alt+G 才接管）",
    )
    parser.add_argument("--no-motion", action="store_true", help="关闭轨迹手势（DTW）")
    parser.add_argument("--no-snap", action="store_true", help="关闭控件磁吸（需要 handmouse[uia]）")
    parser.add_argument(
        "--multi-monitor",
        action="store_true",
        help="按虚拟桌面映射坐标（覆盖 config.multi_monitor）",
    )
    parser.add_argument(
        "--virtual-desk",
        action="store_true",
        help="实验：单屏也走 MOUSEEVENTF_VIRTUALDESK，用于多屏机器上验证",
    )
    parser.add_argument(
        "--margin",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="活动区左右/上下内缩比例，覆盖 config.pointer_margin_*",
    )
    parser.add_argument(
        "--diagnostic-seconds",
        type=float,
        default=0.0,
        help="运行指定秒数后打印指标并退出",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=10.0,
        help="采样心跳间隔秒数，默认 10；短窗口基线（如 30s 无手 gate）调小到 2",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="引导式基线：按固定节拍打印 6 步操作提示（隐含 --auto-control），"
        "未指定 --diagnostic-seconds 时自动在脚本结束后输出诊断并退出",
    )
    parser.add_argument(
        "--list-bindings",
        action="store_true",
        help="打印当前轨迹动作绑定与 LAUNCH 白名单后退出",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.camera is not None:
        config.camera_index = args.camera
    if args.margin is not None:
        config.pointer_margin_x, config.pointer_margin_y = args.margin
    if args.multi_monitor:
        config.multi_monitor = True
    if args.virtual_desk:
        config.virtual_desk_experimental = True
    if args.no_snap:
        config.cursor_snap_enabled = False
    if args.no_motion:
        config.motion_gestures_enabled = False
    config.validate()
    return config


def _pinch_ratio(detection) -> float | None:
    """从最新检测里现算捏合比，只用于调试窗口显示。"""

    if detection is None or not detection.observations:
        return None
    points = detection.observations[0].landmarks
    if len(points) != 21:
        return None
    palm_width = max(hypot(points[5].x - points[17].x, points[5].y - points[17].y), 1e-4)
    return hypot(points[4].x - points[8].x, points[4].y - points[8].y) / palm_width


def _print_bindings(config: AppConfig) -> None:
    print(f"轨迹动作绑定（画完形状后捏合确认才执行）：")
    for name, spec in config.motion_bindings.items():
        print(f"  {name:<8} {spec['kind']:<7} → {spec['target']}")
    print("LAUNCH 白名单（别名 → 命令）：")
    for alias, command in config.launch_whitelist.items():
        print(f"  {alias:<10} → {command}")


def main(argv: list[str] | None = None) -> int:
    # 输出被重定向到日志文件时，stdout 会被 Python 改成块缓冲：
    # 环境指纹和 10s 采样心跳会卡在缓冲区里直到进程退出，"当场可读"就没有了。
    # logging 走 stderr（无缓冲）所以 `[状态]` 行一直正常，这里补齐 stdout。
    sys.stdout.reconfigure(line_buffering=True)

    args = _arguments(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    store = ConfigStore(args.config)
    config = _apply_cli_overrides(store.load(), args)

    if args.list_bindings:
        _print_bindings(config)
        return 0

    recorder: guided.FrameRecorder | None = None
    if args.guided:
        steps = guided.build_steps()
        if args.diagnostic_seconds <= 0:
            # 脚本播完后留 2s 让最后一帧指标落盘，再自动输出诊断退出。
            args.diagnostic_seconds = guided.total_seconds(steps) + 2.0
        print(f"[引导] 共 {len(steps)} 个节拍，约 {guided.total_seconds(steps):.0f} 秒；"
              "请只跟随 [步骤] 提示操作，中途不要碰键盘。")
        recorder = guided.FrameRecorder(
            PROJECT_ROOT / "logs" / f"frames-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        guided.attach_event_trigger(recorder)
        print(f"[取证] 事件抓帧已开启：[状态]/[注入]/[轨迹]/节拍边界自动存图到 {recorder.directory}")

    screen_map = ScreenMap(
        multi_monitor=config.multi_monitor,
        virtual_desk_experimental=config.virtual_desk_experimental,
    )
    safe_mode = bool(args.safe_mode) or sys.platform != "win32"

    try:
        backend = NullMouseBackend(screen_map) if safe_mode else Win32MouseBackend(config, screen_map)
    except RuntimeError as error:
        print(f"平台不受支持：{error}")
        return 2

    mouse = MouseController(backend)
    dispatcher = LaunchDispatcher(config, dry_run=safe_mode)
    pipeline = GesturePipeline(config, mouse, dispatcher=dispatcher)

    state = SimpleNamespace(
        quit=False,
        toggle=False,
        emergency=False,
        started=False,
    )
    hotkeys = HotkeyMonitor(
        lambda: setattr(state, "toggle", True),
        lambda: setattr(state, "emergency", True),
        lambda: setattr(state, "quit", True),
    )
    hotkeys.start()

    print(f"HandMouse {__version__}  后端={type(backend).__name__}  {screen_map.describe()}")
    print(f"配置={store.path}  安全模式={'开' if safe_mode else '关'}")
    if not safe_mode:
        store.save(config)
    print(_environment_line(config, safe_mode))
    _print_bindings(config)
    print("按键：M 暂停/恢复(接管) | ESC/Q 退出 | Ctrl+Alt+G 接管 | Ctrl+Alt+Esc 急停 | Ctrl+Alt+Q 退出")

    if not pipeline.start():
        print("流水线启动失败，详见上方日志。")
        hotkeys.stop()
        mouse.shutdown()
        return 3

    state.started = True
    # 引导模式必须处于接管态，否则引擎只报 paused，看不到 pointer/lost 迁移。
    engine_started_paused = not args.auto_control and not args.guided
    if engine_started_paused:
        pipeline.set_paused(True)
        print("默认未接管鼠标：按 M 或 Ctrl+Alt+G 开始接管。")
    if args.guided:
        guided.start_prompter(
            pipeline.stop_event,
            on_step=lambda index, _prompt: recorder.request(f"步骤-{index:02d}"),
        )

    scale = system_dpi_scale()
    deadline = (
        time.monotonic() + args.diagnostic_seconds if args.diagnostic_seconds > 0 else None
    )
    next_heartbeat = time.monotonic() + max(1.0, args.sample_seconds)
    exit_code = 0
    try:
        while not state.quit:
            _apply_pending_hotkeys(state, pipeline)
            if time.monotonic() >= next_heartbeat:
                _print_heartbeat(pipeline.snapshot())
                next_heartbeat = time.monotonic() + max(1.0, args.sample_seconds)
            frame_packet, detection = pipeline.get_preview()
            if frame_packet is None:
                time.sleep(0.01)
            else:
                snapshot = pipeline.snapshot()
                view = debug_view.build_view(
                    frame_packet.frame,
                    snapshot,
                    detection,
                    pipeline.motion_outcome,
                    screen_map,
                    config,
                    _pinch_ratio(detection),
                )
                if recorder is not None:
                    recorder.capture(view)
                key = debug_view.present(debug_view.WINDOW_NAME, view, scale)
                if not _handle_key(key, state, pipeline, config):
                    break

            if deadline is not None and time.monotonic() >= deadline:
                _print_diagnostics(pipeline.snapshot())
                break
    except KeyboardInterrupt:
        print("\n收到中断，正在安全退出。")
    finally:
        pipeline.stop()
        hotkeys.stop()
        mouse.shutdown()
        debug_view.close()

    if recorder is not None:
        print(f"[取证] 共保存 {recorder.saved} 帧到 {recorder.directory}")
    print(f"退出。DPI={dpi_state()}")
    return exit_code


def _apply_pending_hotkeys(state: SimpleNamespace, pipeline: GesturePipeline) -> None:
    if state.emergency:
        state.emergency = False
        print("[急停] 已释放按键并暂停控制。")
        pipeline.emergency_stop()
    if state.toggle:
        state.toggle = False
        pipeline.toggle_paused()
        print(f"[控制] {'已接管' if not pipeline.engine.paused else '已暂停'}")


def _handle_key(
    key: int,
    state: SimpleNamespace,
    pipeline: GesturePipeline,
    config: AppConfig,
) -> bool:
    """返回 False 表示请求退出。"""

    if key in (-1, 255):
        return True
    if key in (27, ord("q"), ord("Q")):
        return False
    if key in (ord("m"), ord("M")):
        pipeline.toggle_paused()
        print(f"[控制] {'已接管' if not pipeline.engine.paused else '已暂停'}")
    elif key in (ord("r"), ord("R")):
        pipeline.motion_engine.reset()
        pipeline.engine.filter.reset()
        print("[状态] 已重置滤波器与轨迹手势状态")
    return True


def _environment_line(config: AppConfig, safe_mode: bool) -> str:
    """环境指纹：多轮验证靠它确认"同一环境"，并把日志绑到具体的 commit 与配置。

    哈希的是**生效配置**（含 CLI 覆盖），不是磁盘上那份文件——否则 ``--margin``
    这类覆盖改了环境却对不上指纹。
    """

    digest = hashlib.sha256(
        json.dumps(asdict(config), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    commit = "none"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        commit = result.stdout.strip() or "none"
    except (OSError, subprocess.SubprocessError):
        commit = "none"  # 冻结成 EXE 后没有 git，不影响运行
    return (
        f"[环境] commit={commit} config_sha256={digest} "
        f"camera={config.camera_index}@{config.camera_width}x{config.camera_height}"
        f"@{config.camera_fps}fps mirror={config.mirror} "
        f"det={config.min_detection_confidence} pres={config.min_presence_confidence} "
        f"track={config.min_tracking_confidence} "
        f"motion={config.motion_gestures_enabled} snap={config.cursor_snap_enabled} "
        f"safe={safe_mode}"
    )


def _metrics_text(snapshot) -> str:
    metrics = snapshot.metrics
    run_rate = (
        100.0 * metrics.hand_packets_total / metrics.packets_total
        if metrics.packets_total
        else 0.0
    )
    return (
        "capture_fps=%.1f inference_fps=%.1f latency_ms=%.1f latency_p50_ms=%.1f "
        "latency_p95_ms=%.1f hand_rate_3s=%.1f hand_rate_run=%.1f dropped_frames=%d "
        "cpu_percent=%.1f memory_mb=%.1f error=%s"
        % (
            metrics.capture_fps,
            metrics.inference_fps,
            metrics.latency_ms,
            metrics.latency_p50_ms,
            metrics.latency_p95_ms,
            metrics.hand_detection_rate,
            run_rate,
            metrics.dropped_frames,
            metrics.cpu_percent,
            metrics.memory_mb,
            snapshot.camera_error or "none",
        )
    )


def _print_diagnostics(snapshot) -> None:
    print(f"诊断完成 {_metrics_text(snapshot)}")


def _print_heartbeat(snapshot) -> None:
    """每 10 秒一行采样：让"这一轮画面里到底有没有手"当场可读。

    不这样做的话，只能等整轮结束才拿到 ``hand_detection_rate``，而"有没有手"
    恰好是判断测量是否干净的**唯一**依据（没手时必须是 0.0）。
    """

    print(
        f"[采样] gesture={snapshot.gesture.value} paused={snapshot.paused} "
        f"{_metrics_text(snapshot)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
