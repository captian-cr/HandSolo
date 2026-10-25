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

from handmouse import (
    APP_NAME,
    APP_TAGLINE,
    __version__,
    debug_view,
    guided,
    intro,
    snap_ui,
    state_indicator,
    window_ball,
)
from handmouse.config import PROJECT_ROOT, AppConfig, ConfigStore
from handmouse.hotkeys import HotkeyMonitor
from handmouse.mouse import MouseController, NullMouseBackend, Win32MouseBackend
from handmouse.pipeline import GesturePipeline
from handmouse.screen_map import ScreenMap, dpi_state, system_dpi_scale

LOGGER = logging.getLogger(__name__)

#: 「手出画」时圆环还留多久（秒）。留太久会挡住他正在看的东西，
#: 太短又来不及看到——1.2 秒足够注意到，又不会赖在屏幕上。
_LOST_RING_GRACE_S = 1.2

#: 这几种状态下**没有可跟随的东西**（"手出画"光标不动、"已暂停"根本不控制），
#: 圆环只亮一下把原因交代清楚，然后收走。否则开机自启时屏幕正中会永久
#: 挂一个灰圈，看着像程序卡住了。
_TRANSIENT_GESTURES = frozenset({"lost", "paused"})


def _ring_visibility(
    gesture: str, since: float | None, now: float,
) -> tuple[bool, float | None]:
    """这一帧圆环要不要显示，以及更新后的"暂态起点"。

    ``since`` 是该暂态第一次出现的时刻（``None`` = 刚开始）。
    返回值第二项要写回调用方，否则宽限期永远从 0 起算。
    """

    if gesture not in _TRANSIENT_GESTURES:
        return True, None
    if since is None:
        return True, now
    return (now - since) <= _LOST_RING_GRACE_S, since


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handmouse",
        description=f"{APP_NAME} {__version__} —— {APP_TAGLINE}（引擎层移植自 Vimoose，MIT）",
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
        "--no-preview",
        action="store_true",
        help="不显示预览窗（轻量后台运行）：省掉每帧的 HUD 绘制与 imshow，"
        "状态全靠光标旁的小圆环；全局热键 Ctrl+Alt+G/Esc/Q 照常可用",
    )
    parser.add_argument(
        "--onboarding",
        action="store_true",
        help="首次设置向导：先在画面上标点采集你的舒适位置与习惯速率，"
        "再把每种手势带你过一遍并录成你自己的样本库",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="只重做标定（3 个点 + 习惯速率），不录手势、不动已有样本库",
    )
    parser.add_argument(
        "--replay-intro",
        action="store_true",
        help="重放开屏动效（默认只在首次进入控制态时播一次）",
    )
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
        "--infer-width", type=int, default=None, metavar="PX",
        help="推理输入限宽（像素），0=原尺寸。实测 640→320 只提升 13%%："
             "MediaPipe 内部会把输入缩放到固定尺寸，所以外面缩几乎无效。"
             "保留此开关仅用于实验，不建议开。",
    )
    parser.add_argument(
        "--fast-inference", action="store_true",
        help="等价于 --infer-width 320。实测提速有限（7.5→8.5fps），"
             "因为瓶颈在模型本身而非输入分辨率。",
    )
    parser.add_argument(
        "--sign-ref",
        action="store_true",
        help="掌/背朝向符号参照：4 姿势（右手掌/右手背/左手掌/左手背）× 5s"
        " + 3 个 4s 换姿势间隙的定点序列，采集 handedness/nz_unit 参照样本。"
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
    if args.infer_width is not None:
        config.inference_width = max(0, args.infer_width)
    elif args.fast_inference:
        config.inference_width = 320
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
    # ⚠️ pythonw.exe 启动时 sys.stdout 是 None，直接调 reconfigure 会 AttributeError。
    if sys.stdout is not None:
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

    # ---- 首次设置向导（不接管鼠标，单独一条路径）----
    if getattr(args, "onboarding", False):
        from handmouse.onboarding import run_onboarding

        return run_onboarding(config)

    # 只重标定：换了摄像头 / 手的位置变了，但不想把七种手势再录一遍
    if getattr(args, "recalibrate", False):
        from handmouse.onboarding import run_calibration_only

        return run_calibration_only(config)

    # 还没录过自己的习惯：**只提醒一次**（用户原话：「采集只采集一次啊，
    # 不要每次开启都采集一遍」——每次启动都弹会让人以为要反复采集）。
    # 用户不做就不再打扰；想再看到提示，把 config.json 里的
    # onboarding_reminded 改回 false。
    if not config.onboarding_done and not config.onboarding_reminded:
        config.onboarding_reminded = True
        print()
        print("!" * 62)
        print("  还 没 录 过 你 自 己 的 手 势 习 惯")
        print("  现在用的阈值是作者按他自己的手标定的，换一个人会明显不跟手。")
        print("  花两分钟录一遍你自己的（**只需做一次**，之后不再提醒）：")
        print("      python -m handmouse --onboarding")
        print("  （或者：双击 「HandSolo（首次设置）.bat」）")
        print("!" * 62)
        print()

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
        print(f"[符号参照] 共 4 姿势（每段保持 5s）+ 3 个 4s 换姿势间隙 = {total:.0f}s，safe-mode 已开启。"
              "先过取景预检，随后按提示操作：保持段摆好姿势后不动 5s；"
              "间隙 4 秒务必按顺序——先完全出画、停住等状态 lost（≥200ms，引擎关闸），"
              "在画外翻完手/换完手、摆好新姿势后再举回画面，举回后不要再翻。")

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
        #: 预览窗当前是否在显示（运行中可切，见 Ctrl+Alt+P / 窗口内 P）
        preview_on=not getattr(args, "no_preview", False),
        preview_toggle=False,
    )
    hotkeys = HotkeyMonitor(
        lambda: setattr(state, "toggle", True),
        lambda: setattr(state, "emergency", True),
        lambda: setattr(state, "quit", True),
        lambda: setattr(state, "preview_toggle", True),
    )
    hotkeys.start()

    print(f"{APP_NAME} {__version__}  后端={type(backend).__name__}  {screen_map.describe()}")
    print(f"配置={store.path}  安全模式={'开' if safe_mode else '关'}")
    if not safe_mode:
        store.save(config)
    print(_environment_line(config, safe_mode))
    print("按键：M 暂停/恢复(接管) | ESC/Q 退出 | Ctrl+Alt+G 接管 | Ctrl+Alt+Esc 急停 | Ctrl+Alt+Q 退出")

    # 窗口底部悬浮球：**必须在 pipeline.start() 之前建好并注入**。
    # 控制线程在 start() 里构造时就取走了 `pipeline.ball` 这个引用，
    # 之后再赋值它看不到（2026-10-22 踩过：球画出来了但拉不动）。
    # 绘制层建不出来（无显示 / 透明穿透失败）就整个跳过，控制器也不装，
    # 免得出现"看不见球却能拉"的鬼现象。
    ball: window_ball.WindowBallController | None = None
    ball_overlay: window_ball.WindowBallOverlay | None = None
    if getattr(config, "window_ball_enabled", True):
        try:
            ball = window_ball.WindowBallController(enabled=True)
            ball_overlay = window_ball.WindowBallOverlay(ball)
            if not ball_overlay.show():
                ball_overlay = None
                ball = None
            else:
                pipeline.ball = ball
                print("[悬浮球] 已启用：应用底部白色圆球（左右拉扩边、上下拉缩放、松手回弹）")
        except Exception as exc:
            print(f"[悬浮球] 未能创建悬浮球层（{exc}），已跳过。")
            ball = None
            ball_overlay = None

    if not pipeline.start():
        print("流水线启动失败，详见上方日志。")
        hotkeys.stop()
        mouse.shutdown()
        return 3

    # 吸附光环：全屏透明层，命中目标时浮一圈动态彩色环。
    # 建不出来（无显示环境 / 透明穿透设置失败）就安静放弃，绝不挡住桌面点击。
    halo: snap_ui.SnapHaloOverlay | None = None
    if getattr(config, "snap_halo_enabled", True):
        try:
            halo = snap_ui.SnapHaloOverlay()
            if not halo.show():
                halo = None
        except Exception as exc:
            print(f"[光环] 未能创建吸附提示层（{exc}），已跳过。")
            halo = None

    # 状态颜色指示器：没有预览窗时唯一的状态反馈。建不出来就安静跳过。
    indicator: state_indicator.StateIndicator | None = None
    if getattr(config, "state_indicator_enabled", True):
        try:
            indicator = state_indicator.StateIndicator()
            if not indicator.show():
                indicator = None
        except Exception as exc:
            print(f"[指示器] 未能创建状态圆环（{exc}），已跳过。")
            indicator = None
    last_indicator_at = 0.0
    #: 「手出画 / 已暂停」这种暂态是从什么时候开始的（用来先亮圆环、再收走）。
    ring_since: float | None = None
    last_ball_prune_at = 0.0

    state.started = True
    # 开屏动效：只在首次进入控制态时播一次；--replay-intro 可强制重放。
    intro_player: intro.IntroPlayer | None = None
    intro_pending = bool(getattr(args, "replay_intro", False)) or intro.should_play()
    if getattr(args, "replay_intro", False):
        intro.mark_played()
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
    # 预览窗：轻量部署（开机自启 / 桌面双击）不需要它——每帧 build_view + imshow
    # 是整个主循环里最贵的一段，而且窗口会自己弹到最前抢焦点。
    preview_on = not getattr(args, "no_preview", False)
    if not preview_on:
        print(
            "预览窗=关（轻量后台模式）。状态看光标旁的小圆环；"
            "需要看画面时改用「启动%s.bat」。" % APP_NAME,
        )
    # gate-diag 的 deadline 在预检通过那一刻才起算（预检不计时）。
    deadline = (
        time.monotonic() + args.diagnostic_seconds
        if args.diagnostic_seconds > 0 and gate_diag is None
        else None
    )
    next_heartbeat = time.monotonic() + max(1.0, args.sample_seconds)
    exit_code = 0
    # 渲染去重：FramePacket.sequence 是采集线程的全局帧号，采集 30fps。
    # 主循环原本靠 cv2.waitKey(1) 自然节流到约 113 次/秒，其中约 3/4 的重绘
    # 是把同一帧重复画一遍 + 重复 imshow。这既白烧 CPU，又持续迫使 Windows DWM
    # 重绘窗口、把独显从 P8 拉到 P0（实测 2295MHz / 16.8W）。
    #
    # 这里按 sequence 变化判新帧：**无新帧时跳过 build_view/imshow，但仍然走
    # waitKey 喂按键**，所以 M/R/ESC/Q 的响应延迟不受限速影响。
    # 注意不能用 sleep 限速——那会把按键延迟一起拖慢。
    last_rendered_sequence = -1
    # 按键轮询的最小间隔。waitKey(1) 实测阻塞 13.9ms（约 72Hz），
    # 空转时保持这个节奏就够了；有新帧时按采集节奏走。
    try:
        while not state.quit:
            if intro_player is not None:
                # 播开屏期间主循环让位：只推进动画，不跑摄像头与按键。
                # 三秒多而已，热键（Ctrl+Alt+Esc）仍然有效。
                if intro_player.tick():
                    time.sleep(0.01)
                    continue
                intro_player.close()
                intro_player = None
                intro.mark_played()
                continue
            _apply_pending_hotkeys(state, pipeline)
            preview_on = state.preview_on
            if time.monotonic() >= next_heartbeat:
                _print_heartbeat(pipeline.snapshot())
                next_heartbeat = time.monotonic() + max(1.0, args.sample_seconds)
            frame_packet, detection = pipeline.get_preview()
            if frame_packet is None:
                time.sleep(0.01)
            else:
                # ---- 关键路径：新帧才重绘；旧帧只喂按键 ----
                is_new_frame = frame_packet.sequence != last_rendered_sequence
                if not is_new_frame:
                    if preview_on:
                        # 画面没变，但按键必须继续读，否则窗口会"卡住"按不动。
                        if not _handle_key(
                            debug_view.present(debug_view.WINDOW_NAME, None, scale),
                            state, pipeline, config,
                        ):
                            break
                    else:
                        # 没有预览窗就没有 waitKey 顺手帮我们节流，自己让一点 CPU。
                        time.sleep(0.005)
                    continue

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
                if intro_pending and snapshot.gesture.value == "pointer":
                    intro_pending = False
                    player = intro.IntroPlayer()
                    if player.start():
                        intro_player = player
                        continue
                    intro.mark_played()
                view = (
                    debug_view.build_view(
                        frame_packet.frame,
                        snapshot,
                        detection,
                        screen_map,
                        config,
                        _pinch_ratio(detection),
                        banner=banner,
                    )
                    if preview_on
                    else None
                )
                if halo is not None and not halo.closed:
                    halo.update(snapshot.snap_rect, snapshot.snap_target_name or "")
                    halo.refresh()
                if ball is not None and ball_overlay is not None and not ball_overlay.closed:
                    # 光标位置用引擎自己记的归一化坐标换算成屏幕像素：
                    # 球心是按窗口矩形算的像素坐标，两边必须同一坐标系。
                    cx, cy = pipeline.engine.cursor_position
                    bx, by = screen_map.to_pixels(cx, cy)
                    # 14fps 左右足够：球只在淡入淡出和回弹时有变化。
                    ball.update(time.monotonic_ns(), (bx, by))
                    # 每 2 秒清一次僵尸窗口，免得球越积越多。
                    if time.monotonic() - last_ball_prune_at >= 2.0:
                        last_ball_prune_at = time.monotonic()
                        ball.prune()
                        # 顺带把球层提到最前：吸附光环/状态圆环是**后**创建的，
                        # 同为 topmost 时会盖在球层上面。球在窗口下方那块通常
                        # 不重叠，但重叠一次就"球不见了"，代价太大。
                        try:
                            ball_overlay.root.lift()
                        except Exception:
                            pass
                    ball_overlay.refresh()
                    # HUD 诊断：看不见球时靠这行直接看出卡在哪一步
                    # （没有前台窗口 / 透明度是 0 / 球心跑到屏幕外）。
                    track = ball.tracks.get(ball.active_hwnd)
                    if track is None:
                        snapshot.ball_note = "球 hwnd=0（没取到前台窗口）"
                    else:
                        sx, sy = track.center
                        on_screen = 0 <= sx <= ball_overlay.screen_w and 0 <= sy <= ball_overlay.screen_h
                        snapshot.ball_note = (
                            f"球 hwnd={track.hwnd} alpha={track.alpha:.2f} "
                            f"({sx:.0f},{sy:.0f}) {'屏内' if on_screen else '屏外!'}"
                            + (" 抓着" if ball.grabbing else "")
                        )
                elif ball_overlay is None:
                    snapshot.ball_note = "球 未启用（叠加层创建失败）"
                if indicator is not None and not indicator.closed:
                    # 指示器只需要 ~15fps：圆环跟随光标，太快纯属浪费合成开销。
                    now_i = time.monotonic()
                    if now_i - last_indicator_at >= 0.066:
                        last_indicator_at = now_i
                        cx, cy = pipeline.engine.cursor_position
                        px, py = screen_map.to_pixels(cx, cy)
                        gesture_now = pipeline.engine.current_gesture.value
                        # 「手出画 / 已暂停」没有可跟随的东西：先亮 1.2 秒把原因
                        # 交代清楚，再收走（用户要求：光标没了圆环也跟着消失）。
                        # 没有预览窗时，圆环突然消失和"程序崩了"长得一样，
                        # 所以那 1.2 秒不能省。
                        show_ring, ring_since = _ring_visibility(
                            gesture_now, ring_since, time.monotonic(),
                        )
                        indicator.update(
                            screen_x=px,
                            screen_y=py,
                            gesture=gesture_now,
                            visible=show_ring,
                        )
                        indicator.refresh()
                if recorder is not None and view is not None:
                    recorder.capture(view)
                last_rendered_sequence = frame_packet.sequence
                key = (
                    debug_view.present(debug_view.WINDOW_NAME, view, scale)
                    if preview_on
                    else -1
                )
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
        if intro_player is not None:
            intro_player.close()
        if indicator is not None:
            indicator.close()
        if halo is not None:
            halo.close()
        debug_view.close()

    if recorder is not None:
        print(f"[取证] 共保存 {recorder.saved} 帧到 {recorder.directory}")
    print(f"退出。DPI={dpi_state()}")
    return exit_code


def _set_preview(state: SimpleNamespace, want: bool | None) -> bool:
    """开关预览窗。``want=None`` 表示"切一下"。返回切换后的状态。

    🔴 关窗必须走 `debug_view.close()`：它会把内部的"窗口建过没有"标志一起清掉，
    否则下次 `imshow` 会拿一个已经销毁的窗口去画，Windows 上表现为窗口再也不出现。
    """

    target = (not state.preview_on) if want is None else bool(want)
    if target == state.preview_on:
        return state.preview_on
    state.preview_on = target
    if target:
        print("[预览] 已打开（Ctrl+Alt+P 或窗口内按 P 可再关掉）")
    else:
        debug_view.close(debug_view.WINDOW_NAME)
        print("[预览] 已关闭，状态改看光标旁的小圆环；Ctrl+Alt+P 可再打开")
    return target


def _apply_pending_hotkeys(state: SimpleNamespace, pipeline: GesturePipeline) -> None:
    if state.emergency:
        state.emergency = False
        print("[急停] 已释放按键并暂停控制。")
        pipeline.emergency_stop()
    if state.toggle:
        state.toggle = False
        pipeline.toggle_paused()
        print(f"[控制] {'已接管' if not pipeline.engine.paused else '已暂停'}")
    if state.preview_toggle:
        state.preview_toggle = False
        _set_preview(state, None)


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
    elif key in (ord("p"), ord("P")):
        _set_preview(state, None)
    elif key in (ord("r"), ord("R")):
        pipeline.engine.filter.reset()
        print("[状态] 已重置滤波器")
    elif key in (ord("+"), ord("=")):
        config.cursor_speed_multiplier = min(
            10.0, round(config.cursor_speed_multiplier + 0.2, 2),
        )
        print(f"[摇杆] 速度倍率 → {config.cursor_speed_multiplier}x")
    elif key in (ord("]"), ord("}")):
        _nudge_margins(config, +0.02, pipeline)
        _print_feel(config, "更灵敏")
    elif key in (ord("["), ord("{")):
        _nudge_margins(config, -0.02, pipeline)
        _print_feel(config, "更精细")
    elif key == ord("."):
        # 更跟手：降低静止平滑（min_cutoff 升高），代价是抖。
        config.pointer_stability = max(
            0.0, round(config.pointer_stability - 0.04, 2),
        )
        pipeline.engine.update_config(config)
        _print_feel(config, "静止更跟手")
    elif key == ord(","):
        # 更稳：提高静止平滑，代价是略钝。
        config.pointer_stability = min(
            1.0, round(config.pointer_stability + 0.04, 2),
        )
        pipeline.engine.update_config(config)
        _print_feel(config, "静止更稳")
    elif key == ord(">"):
        # 大动作更跟手：beta 升高 —— 快速移动时滤波器 cutoff 抬得更高，
        # 光标更贴手，但也更容易冲过头。
        config.pointer_response = min(
            1.0, round(config.pointer_response + 0.04, 2),
        )
        pipeline.engine.update_config(config)
        _print_feel(config, "大动作更跟手")
    elif key == ord("<"):
        # 大动作更稳：beta 降低 —— 手大幅快速移动时滤波仍然很强，
        # 光标不会跟着甩出去，代价是快速掠过时略微拖尾。
        # 这就是"大幅转动时降低灵敏度、保持稳定"的那个旋钮。
        config.pointer_response = max(
            0.0, round(config.pointer_response - 0.04, 2),
        )
        pipeline.engine.update_config(config)
        _print_feel(config, "大动作更稳")
    elif key in (ord("-"), ord("_")):
        config.cursor_speed_multiplier = max(
            0.2, round(config.cursor_speed_multiplier - 0.2, 2),
        )
        print(f"[摇杆] 速度倍率 → {config.cursor_speed_multiplier}x")
    return True


def _nudge_margins(config: AppConfig, delta: float, pipeline: GesturePipeline) -> None:
    """四边内缩同步增减（保持非对称形状，只改整体灵敏度）。

    走 `effective_margins()`：如果配置里用的是四边独立值，就直接改那四个；
    否则把对称值展开成四个再改，之后统一按非对称处理。
    """

    left, top, right, bottom = config.effective_margins()
    left = max(0.0, min(0.35, left + delta))
    top = max(0.0, min(0.35, top + delta))
    right = max(0.0, min(0.35, right + delta))
    bottom = max(0.0, min(0.35, bottom + delta))
    # 任一轴加起来不能吃掉整个画面。
    if top + bottom > 0.9:
        bottom = max(0.0, 0.9 - top)
    if left + right > 0.9:
        right = max(0.0, 0.9 - left)
    config.pointer_margin_left = left
    config.pointer_margin_top = top
    config.pointer_margin_right = right
    config.pointer_margin_bottom = bottom
    config.validate()
    pipeline.engine.update_config(config)


def _print_feel(config: AppConfig, label: str) -> None:
    """手感调整后的回执：把两个旋钮的当前值换算成能感知的量。"""

    left, top, right, bottom = config.effective_margins()
    travel_x = max(0.01, 1.0 - left - right)
    travel_y = max(0.01, 1.0 - top - bottom)
    # 两个平滑旋钮的物理含义（One Euro）：
    #   stability 越高 → min_cutoff 越低 → 手停住时越不抖
    #   response  越高 → beta 越高   → 快速移动时越贴手（也越容易冲过头）
    min_cutoff = 3.0 - 2.2 * config.pointer_stability
    beta = 0.35 + 1.55 * config.pointer_response
    print(
        f"[手感] {label} → 内缩 上{top:.2f} 下{bottom:.2f} "
        f"左{left:.2f} 右{right:.2f}"
        f"（手行程 {travel_x:.0%}×{travel_y:.0%}）  "
        f"静止 stability={config.pointer_stability:.2f}(cutoff {min_cutoff:.2f})  "
        f"大动作 response={config.pointer_response:.2f}(beta {beta:.2f})",
    )


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
        f"traj_correction={'on' if config.enable_trajectory_correction else 'off'} "
        f"wrist_dist_thresh={config.trajectory_correction_wrist_dist:.2f} "
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
        "cpu_percent=%.1f memory_mb=%.1f thumb=%.2f error=%s"
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
            metrics.thumb_spread_percent / 100.0,
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
