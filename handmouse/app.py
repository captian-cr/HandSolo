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
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from math import hypot
from pathlib import Path
from types import SimpleNamespace

from handmouse import __version__, debug_view, guided
from handmouse.config import PROJECT_ROOT, AppConfig, ConfigStore
from handmouse.hotkeys import HotkeyMonitor
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
        "--gate-diag",
        action="store_true",
        help="进入门定性定点：30s 内 6 段固定姿势序列，用于定位进入门判定链断点。"
        "隐含 --diagnostic-seconds 30 和 --safe-mode。"
        "需同时设置环境变量 HANDMOUSE_VERBOSE_GATE=1 才能启用诊断日志。",
    )
    parser.add_argument(
        "--sign-ref",
        action="store_true",
        help="掌/背朝向符号参照：4 姿势（右手掌/右手背/左手掌/左手背）× 5s"
        " + 3 个 2s 换姿势间隙的定点序列，采集 handedness/cross/sin 参照样本。"
        "隐含 --safe-mode。"
        "需同时设置环境变量 HANDMOUSE_SIGN_REF=1 才会输出诊断日志"
        "（该开关只加诊断日志、不改任何判定）。",
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


def _first_palm_width(detection) -> float:
    """第一只手的掌宽（landmark 5→17），供 gate-diag 预检判断手是否够近。

    定义与引擎进入门一致（x 方向归一化间距，见 gestures._palm_width），
    这样 HUD 预检值与诊断日志 geom 行、ENTRY_PALM_WIDTH_MIN 直接可比。
    """

    if detection is None or not detection.observations:
        return 0.0
    points = detection.observations[0].landmarks
    if len(points) != 21:
        return 0.0
    return abs(points[17].x - points[5].x)


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

    # 进入门诊断开关：环境变量控制，跨模块生效。
    # 参照模式开关（仅日志）：与 VERBOSE_GATE 共用一个 FileHandler，但用独立
    # 文件名前缀归档，避免两轮相互覆盖。
    verbose_gate = os.environ.get("HANDMOUSE_VERBOSE_GATE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    sign_ref = os.environ.get("HANDMOUSE_SIGN_REF", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    diag_logger: logging.Logger | None = None
    if verbose_gate or sign_ref:
        diag_dir = PROJECT_ROOT / "logs"
        diag_dir.mkdir(parents=True, exist_ok=True)
        prefix = "sign-ref" if sign_ref else "gate-diag"
        diag_path = diag_dir / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        diag_logger = logging.getLogger("handmouse.gate-diag")
        diag_logger.setLevel(logging.INFO)
        diag_logger.propagate = False
        handler = logging.FileHandler(str(diag_path), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        diag_logger.addHandler(handler)
        flag = "HANDMOUSE_SIGN_REF=1" if sign_ref else "HANDMOUSE_VERBOSE_GATE=1"
        print(f"[诊断] {flag} 已启用，诊断日志 → {diag_path}")

    store = ConfigStore(args.config)
    config = _apply_cli_overrides(store.load(), args)

    # gate-diag 启动头：把本轮的关键判定前提（镜像约定 + 朝向幅度门槛）钉进
    # 诊断日志文件头。① 修复后 expected_sign 由 mirror×handedness 派生，逐帧
    # cross 行虽也带 mirror，但日志按轮单独命名归档、隔轮回看时需要一个必然
    # 存在的运行头，避免"这一轮跑的是哪种镜像约定"无法自证。
    if diag_logger is not None and args.gate_diag:
        diag_logger.info("%s", _gate_diag_run_header(config))
    # sign-ref 启动头：同样钉死镜像约定，并显式声明"仅参照、不改判定"。
    if diag_logger is not None and args.sign_ref:
        diag_logger.info("%s", _sign_ref_run_header(config))

    recorder: guided.FrameRecorder | None = None
    # 进入门定点序列：gate-diag（6 段×5s）与 sign-ref（4 姿势×3s）共用
    # GateDiagController 的取景预检 + 节拍驱动，只是 steps 不同。
    diag_steps: list[tuple[float, str]] | None = None
    if args.gate_diag:
        diag_steps = guided.build_gate_diag_steps()
        args.diagnostic_seconds = 30.0
        args.safe_mode = True
        print(f"[进入门诊断] 共 {len(diag_steps)} 段 × 5s = 30s，safe-mode 已开启。"
              "先过取景预检（连续 30 帧检到手且 palm_width≥0.15）才开始计时，"
              "请按调试窗 [预检] 提示把手放到画面中央、靠近镜头。")
    elif args.sign_ref:
        diag_steps = guided.build_sign_ref_steps()
        total = guided.total_seconds(diag_steps)
        args.diagnostic_seconds = total
        args.safe_mode = True
        if not sign_ref:
            print("[符号参照] 未检测到 HANDMOUSE_SIGN_REF=1：几何日志仍会写，"
                  "但 [sign-ref] cross/sin 参照行不会输出。请以 HANDMOUSE_SIGN_REF=1 启动。")
        print(f"[符号参照] 共 4 姿势（每段保持 5s）+ 3 个 2s 换姿势间隙 = {total:.0f}s，safe-mode 已开启。"
              "先过取景预检，随后按提示操作：保持段摆好姿势后不动 5s，"
              "间隙里把翻手/换手做完再进入下一段（可出画再回来）。")

    # safe_mode 在 gate-diag/sign-ref 强制 safe 之后才定格，故放在这里而非更早。
    safe_mode = bool(args.safe_mode) or sys.platform != "win32"
    # 环境指纹（commit + config_sha256 + 工作区脏标记）落进诊断文件头：取证与跨轮
    # 比对只读这一份结构化日志，它必须能自证"这一轮跑的是哪个 commit、哪份生效
    # 配置"，不能只指望被重定向的 stdout（gatediag-stdout-*.log）。与 stdout 复用
    # 同一 _environment_line，保证两份文件的指纹行逐字一致、可对拍。
    if diag_logger is not None:
        diag_logger.info("%s", _environment_line(config, safe_mode))

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
        print(f"[取证] 事件抓帧已开启：[状态]/[注入]/节拍边界自动存图到 {recorder.directory}")

    screen_map = ScreenMap(
        multi_monitor=config.multi_monitor,
        virtual_desk_experimental=config.virtual_desk_experimental,
    )

    try:
        backend = NullMouseBackend(screen_map) if safe_mode else Win32MouseBackend(config, screen_map)
    except RuntimeError as error:
        print(f"平台不受支持：{error}")
        return 2

    mouse = MouseController(backend)
    pipeline = GesturePipeline(config, mouse)

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
    gate_diag: guided.GateDiagController | None = None
    if args.guided:
        # 第 25 步（H5' 引擎级降级恢复）：自动暂停 3 秒后恢复，不碰相机硬件。
        # 暂停期间应无注入；恢复后手回画面应能重新进入操控态。
        def _step25_pause_resume() -> None:
            pipeline.set_paused(True)

            def _resume() -> None:
                time.sleep(3.0)
                if not pipeline.stop_event.is_set():
                    pipeline.set_paused(False)

            threading.Thread(target=_resume, name="guided-step25-resume", daemon=True).start()

        guided.start_prompter(
            pipeline.stop_event,
            on_step=lambda index, _prompt: recorder.request(f"步骤-{index:02d}"),
            step_actions={25: _step25_pause_resume},
        )
    elif diag_steps is not None:
        # gate-diag / sign-ref 节拍不另起线程：预检/中止/分段都由主循环逐帧
        # 驱动 GateDiagController（时钟注入、可单测），避免线程与取景状态脱节。
        gate_diag = guided.GateDiagController(diag_steps)

    scale = system_dpi_scale()
    # gate-diag 的 deadline 在预检通过那一刻才起算（预检不计时）。
    deadline = (
        time.monotonic() + args.diagnostic_seconds
        if args.diagnostic_seconds > 0 and gate_diag is None
        else None
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
                banner = None
                if gate_diag is not None:
                    hand_count = (
                        len(detection.observations) if detection is not None else 0
                    )
                    aborted = False
                    for event in gate_diag.observe(
                        time.monotonic(),
                        hand_count,
                        _first_palm_width(detection),
                        float(frame_packet.frame.std()),
                    ):
                        print(event.line)
                        if event.kind == "precheck_pass":
                            # 30s 序列 +2s 收尾后输出诊断退出；预检耗时不计入。
                            deadline = (
                                time.monotonic() + args.diagnostic_seconds + 2.0
                            )
                        elif event.kind == "abort":
                            state.quit = True
                            aborted = True
                    if aborted:
                        exit_code = 4
                    banner = (
                        gate_diag.hud_text(),
                        debug_view.RED if gate_diag.hud_alert else debug_view.GREEN,
                    )
                view = debug_view.build_view(
                    frame_packet.frame,
                    snapshot,
                    detection,
                    screen_map,
                    config,
                    _pinch_ratio(detection),
                    banner=banner,
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
        pipeline.engine.filter.reset()
        print("[状态] 已重置滤波器")
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
        f"[环境] commit={commit}{_worktree_mark()} config_sha256={digest} "
        f"camera={config.camera_index}@{config.camera_width}x{config.camera_height}"
        f"@{config.camera_fps}fps mirror={config.mirror} "
        f"det={config.min_detection_confidence} pres={config.min_presence_confidence} "
        f"track={config.min_tracking_confidence} "
        f"snap={config.cursor_snap_enabled} "
        f"safe={safe_mode}"
    )


def _worktree_mark() -> str:
    """工作区有未提交改动时打 ``+dirty``，避免"假指纹"。

    commit 只说明 HEAD 是谁，不代表跑的代码就是它——r1→r2 之间有人在未提交的
    工作区里改了阈值和检测逻辑，两轮日志却都写着同一个 commit，导致不可比。
    这里把工作区状态一起写进指纹行。

    必须**上报未跟踪文件**：新逻辑写在新文件（尚未 ``git add``）里时，只比
    tracked 改动会漏掉它，留下"hash 相同、代码不同"的盲区。``logs/`` 已在
    ``.gitignore``，普通 untracked 模式不会把它算进来，故无需再关掉上报。
    """

    try:
        result = subprocess.run(
            # 覆盖已跟踪改动 + 未跟踪的新文件；被 gitignore 的 logs/ 不出现。
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""  # 冻结成 EXE 后没有 git
    return "+dirty" if result.stdout.strip() else ""


def _gate_diag_run_header(config: AppConfig) -> str:
    """gate-diag 诊断日志的运行头：记录镜像约定、进入门幅度门与符号锁存参数。

    ``expected_sign`` 由 ``mirror × handedness`` 派生（同一镜像下左/右手符号
    相反），无法用单一标量表示，故此处只记录 ``mirror`` 这个输入变量本身。
    朝向观测量本笔由旧 ``cross``（5→17 × 5→4）换成 ``nz_unit``（0-5-17 三角
    朝向）；``entry_palm_nz_min`` 只拒几何退化帧，符号改由锁存压过渡块
    （``entry_palm_nz_commit`` 参与门 + ``entry_facing_commit_frames`` 帧确认）。
    """

    return (
        f"[gate-diag运行头] mirror={config.mirror} "
        f"entry_palm_nz_min={config.entry_palm_nz_min:.2f} "
        f"entry_palm_nz_commit={config.entry_palm_nz_commit:.2f} "
        f"entry_facing_commit_frames={config.entry_facing_commit_frames} "
        "(expected_sign 由 mirror×handedness 派生)"
    )


def _sign_ref_run_header(config: AppConfig) -> str:
    """sign-ref 诊断日志的运行头：钉死镜像约定，并声明本模式不改判定。

    参照行（``[sign-ref] nz_unit``）与 gate-diag 的 ``nz_unit`` 行同源，隔轮
    回看时必须能从文件头自证"这一轮跑的是哪种镜像约定"。同时显式写明本模式
    只加日志、不改判定，避免将来考古把它误读成判定逻辑。观测量口径本笔已由
    ``cross`` 换成 ``nz_unit``，新旧参照日志不可混读。
    """

    return (
        f"[sign-ref运行头] mirror={config.mirror} "
        f"entry_palm_nz_min={config.entry_palm_nz_min:.2f} "
        f"entry_palm_nz_commit={config.entry_palm_nz_commit:.2f} "
        f"entry_facing_commit_frames={config.entry_facing_commit_frames} "
        "（仅参照模式：只加诊断日志、不改任何判定）"
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
