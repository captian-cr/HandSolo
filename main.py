"""手势控制电脑 · 第一阶段验证脚本

验证目标（只验这两件事，先别加状态机）
--------------------------------------
1. 映射精度：食指尖能不能稳定覆盖整块屏幕，特别是四个角。
2. 捏合误触：说话、拿水杯、手放下等日常动作，会不会被误判成捏合。
   脚本实时记录捏合比值的 min/max，并在每次触发瞬间打印比值，
   跑几分钟就能据实数据定下 pinch_on_ratio / pinch_off_ratio。

运行
----
    python main.py                 # 启动（默认不接管鼠标，按 M 才接管）
    python main.py --calibrate     # 只做校准、保存后退出

按键（OpenCV 窗口需获得焦点）
--------------------------
    M      开启 / 关闭鼠标接管
    C      重新校准
    R      重置滤波器与统计
    B      校准时回退一步
    ESC/Q  退出

安全提示
--------
鼠标被接管后 OpenCV 窗口可能失去焦点，此时 ESC 按不到。
脱身办法：握拳保持 0.6 秒 → 光标自动冻结，再点回窗口按 ESC。
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from gesture.calibration import TARGET_NAMES, CalibrationSession
from gesture.config import Config
from gesture.hand_tracker import HandTracker
from gesture.hud import DEFAULT_GRAY, DEFAULT_GREEN, DEFAULT_RED, DEFAULT_WHITE, DEFAULT_YELLOW
from gesture.hud import draw_hud
from gesture.mapping import CALIB_PATH, ScreenMapper
from gesture.pointer import PointerController
from gesture import winmouse

WINDOW = "Gesture Control - Stage 1"


def open_camera(cfg: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={cfg.camera_index}，请检查是否被其他程序占用")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def format_point(point) -> str:
    if point is None:
        return "n/a"
    return f"({point[0]:.0f},{point[1]:.0f})"


class App:
    def __init__(self, cfg: Config, calibrate_only: bool = False) -> None:
        self.cfg = cfg
        self.calibrate_only = calibrate_only

        winmouse.enable_dpi_awareness()
        self.screen_w, self.screen_h = winmouse.screen_size()
        self.display_scale = winmouse.dpi_scale()

        self.cap = open_camera(cfg)
        ok, warmup = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("摄像头读取失败")
        self.frame_h, self.frame_w = warmup.shape[:2]

        self.tracker = HandTracker(
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            index_extend_ratio=cfg.index_extend_ratio,
        )

        self.mapper = ScreenMapper()
        if self.mapper.load():
            if (self.mapper.screen_w, self.mapper.screen_h) != (self.screen_w, self.screen_h):
                print("[校准] 屏幕分辨率与上次校准不一致，需要重新校准。")
                self.mapper.reset()
            else:
                print(f"[校准] 已载入：屏幕 {self.screen_w}x{self.screen_h}，cam_rect={self.mapper.cam_rect}")

        self.pointer = PointerController(cfg, self.mapper)
        self.session = CalibrationSession(cfg, (self.frame_w, self.frame_h))
        self.calibrating = not self.mapper.ready

        self.fps = 0.0
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0

    # ---------------- 校准 ----------------
    def _finish_calibration(self) -> None:
        try:
            self.mapper.calibrate(
                self.session.samples,
                (self.screen_w, self.screen_h),
                cam_rect=self.session.rect,
            )
        except ValueError as exc:
            print(f"[校准] 失败：{exc}")
            self.session.reset()
            return
        self.mapper.save()
        self.calibrating = False
        self.pointer.reset()
        points = np.round(np.asarray(self.session.samples), 1).tolist()
        print(f"[校准] 完成并保存到 {CALIB_PATH}")
        print(f"[校准] 采样点 {points}  屏幕 {self.screen_w}x{self.screen_h}")
        print("[校准] 按 M 开启鼠标接管；先确认四角可达、再确认捏合阈值。")

    # ---------------- HUD ----------------
    def _status_row(self, info: dict) -> tuple[str, tuple]:
        if not self.mapper.ready:
            return "未校准 - 请按提示完成校准", DEFAULT_YELLOW
        if not self.pointer.enabled:
            return "鼠标未接管 - 按 M 开启", DEFAULT_YELLOW
        if info["frozen"]:
            return "握拳暂停中 - 伸直食指即恢复", DEFAULT_YELLOW
        if info["pinched"]:
            return "捏合中（左键按住）", DEFAULT_RED
        return "跟踪中", DEFAULT_GREEN

    def _draw_hud(self, frame: np.ndarray, info: dict) -> None:
        ratio = info["ratio"]
        ratio_text = f"{ratio:.2f}" if ratio is not None else "--"
        if self.pointer.ratio_max > 0:
            range_text = f"{self.pointer.ratio_min:.2f}/{self.pointer.ratio_max:.2f}"
        else:
            range_text = "--"

        status_text, status_color = self._status_row(info)
        rows = [
            (status_text, status_color),
            (
                f"捏合比 {ratio_text}  阈值 on<{self.cfg.pinch_on_ratio:.2f} off>{self.cfg.pinch_off_ratio:.2f}"
                f"  实测 min/max {range_text}",
                DEFAULT_WHITE,
            ),
            (
                f"指向源 {info['pointer_source']}  屏幕 {format_point(info['screen'])}"
                f"  已点击 {self.pointer.clicks}",
                DEFAULT_WHITE,
            ),
            (f"FPS {self.fps:.1f}   M 接管  C 校准  R 重置  ESC 退出", DEFAULT_GRAY),
        ]
        draw_hud(frame, rows)

    # ---------------- 按键 ----------------
    def _handle_key(self, key: int) -> bool:
        if key in (27, ord("q"), ord("Q")):
            return False

        char = chr(key).lower() if 32 <= key < 127 else ""
        if char == "m":
            if not self.mapper.ready:
                print("[提示] 还没有校准，先完成校准才能接管鼠标。")
            else:
                print(f"[鼠标] {'已开启' if self.pointer.toggle() else '已关闭'}接管")
        elif char == "c":
            self.mapper.reset()
            self.session.reset()
            self.pointer.set_enabled(False)
            self.calibrating = True
            print("[校准] 重新开始校准流程")
        elif char == "r":
            if self.calibrating:
                self.session.reset()
                print("[校准] 已重置采样")
            else:
                self.pointer.reset()
                print("[统计] 已重置滤波器与比值统计")
        elif char == "b" and self.calibrating:
            self.session.back()
        return True

    # ---------------- 主循环 ----------------
    def run(self) -> None:
        print(f"摄像头 {self.frame_w}x{self.frame_h}  屏幕 {self.screen_w}x{self.screen_h}"
              f"  显示缩放 {self.display_scale:.2f}")
        print("按键：M 接管 | C 校准 | R 重置 | B 回退 | ESC 退出")
        print("脱身：握拳保持 0.6 秒冻结光标，再点回窗口按 ESC")

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("摄像头读取失败，退出。")
                    break

                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                self._fps_frames += 1
                if self._fps_frames >= 15:
                    self.fps = self._fps_frames / max(now - self._fps_t0, 1e-6)
                    self._fps_t0 = now
                    self._fps_frames = 0

                obs = self.tracker.process(frame)
                if obs is not None:
                    self.tracker.draw_skeleton(frame, obs.points)

                info = self.pointer.update(obs, now)

                if self.calibrating:
                    recorded = self.session.update(info["cam_point"])
                    self.session.draw(frame, info["cam_point"], self.fps)
                    if recorded:
                        print(f"[校准] 已记录 {TARGET_NAMES[self.session.step - 1]} "
                              f"({format_point(info['cam_point'])})")
                    if self.session.done:
                        self._finish_calibration()
                        if self.calibrate_only:
                            break
                else:
                    self._draw_hud(frame, info)

                view = frame
                if self.display_scale > 1.01:
                    view = cv2.resize(
                        frame,
                        None,
                        fx=self.display_scale,
                        fy=self.display_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow(WINDOW, view)

                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        self.pointer.shutdown()
        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()

        print("\n=== 本次运行统计 ===")
        if self.pointer.ratio_max > 0:
            print(f"捏合比值范围：{self.pointer.ratio_min:.3f} ~ {self.pointer.ratio_max:.3f}")
        print(f"触发点击：{self.pointer.clicks} 次")
        if self.pointer.click_ratios:
            ratios = np.asarray(self.pointer.click_ratios)
            print(f"点击瞬间比值：min={ratios.min():.3f} max={ratios.max():.3f} mean={ratios.mean():.3f}")
            print(f"建议 pinch_on_ratio ≈ {ratios.max() * 1.2:.2f}，"
                  f"pinch_off_ratio ≈ {min(ratios.max() * 1.2 + 0.12, 0.9):.2f}")
        else:
            print("建议：先正常做几次捏合，再看这里的推荐阈值。")


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手势控制电脑 · 第一阶段验证脚本")
    parser.add_argument("--calibrate", action="store_true", help="只做校准并保存，然后退出")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = Config.load()
    try:
        app = App(cfg, calibrate_only=args.calibrate)
    except RuntimeError as exc:
        print(f"启动失败：{exc}")
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
