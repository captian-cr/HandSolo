"""引导取证抓帧（FrameRecorder）与节拍回调的测试。"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from handmouse import guided
from handmouse.guided import FrameRecorder, attach_event_trigger


def _frame() -> np.ndarray:
    return np.zeros((60, 80, 3), dtype=np.uint8)


def test_sanitize_keeps_cjk_and_arrow_strips_specials() -> None:
    recorder = FrameRecorder.__new__(FrameRecorder)
    recorder.max_reason = 60
    text = recorder._sanitize("[状态] none(paused=False) → scroll(paused=False)")
    assert " " not in text and "(" not in text and "[" not in text
    assert "状态" in text and "→" in text
    assert len(text) <= 60


def test_capture_requires_request(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path)
    assert recorder.capture(_frame()) is False
    assert recorder.saved == 0
    assert list(tmp_path.iterdir()) == []


def test_capture_writes_jpg_named_with_reason(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path)
    recorder.request("步骤-03")
    assert recorder.capture(_frame()) is True
    files = list(tmp_path.glob("*.jpg"))
    assert len(files) == 1
    assert "步骤-03" in files[0].name
    assert files[0].stat().st_size > 0
    assert recorder.saved == 1


def test_debounce_coalesces_burst(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path, debounce_seconds=10.0)
    recorder.request("[状态] a → b")
    assert recorder.capture(_frame()) is True
    recorder.request("[注入] left_click")
    assert recorder.capture(_frame()) is False  # 仍在防抖窗口内
    assert recorder.saved == 1
    # 事件不丢：原因挂到下一帧
    recorder._last_capture -= 11.0
    assert recorder.capture(_frame()) is True
    names = [p.name for p in tmp_path.glob("*.jpg")]
    assert any("left_click" in name for name in names)


def test_attach_event_trigger_matches_tags(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path)
    handler = attach_event_trigger(recorder)
    logger = logging.getLogger("handmouse.pipeline")
    # 真实运行里 basicConfig(level=INFO) 保证 INFO 可达；测试环境默认 WARNING，
    # 记录会在到达 handler 前被级别过滤，所以临时放开。
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        logger.info("[状态] none(paused=False) → scroll(paused=False)")
        logger.info("[注入] left_click")
        logger.info("普通日志不触发")
        logger.info("[轨迹] state=candidate candidates=W pending=- | 候选 W")
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(handler)
    with recorder._lock:
        pending = list(recorder._pending)
    assert sum("[状态]" in p for p in pending) == 1
    assert sum("[注入]" in p for p in pending) == 1
    assert sum("[轨迹]" in p for p in pending) == 1
    assert all("普通日志" not in p for p in pending)


def test_prompter_invokes_on_step(monkeypatch) -> None:
    monkeypatch.setattr(guided, "build_steps", lambda: [(0.05, "测试节拍")])
    calls: list[tuple[int, str]] = []
    stop = threading.Event()
    thread = guided.start_prompter(stop, on_step=lambda i, p: calls.append((i, p)))
    thread.join(timeout=2.0)
    assert calls == [(1, "测试节拍")]
