"""GateDiagController 单测：取景预检、seg1 自动中止、节拍推进。

时钟全部由 ``now`` 注入，测试不依赖相机/线程/真实时间。
"""

from handmouse.guided import GateDiagController, build_sign_ref_steps, total_seconds


def test_build_sign_ref_steps_shape():
    """参照序列：4 姿势保持段 × 3s，总 12s。"""

    steps = build_sign_ref_steps()
    holds = [s for s in steps if s[1].startswith("参照")]
    assert len(holds) == 4
    assert all(duration == 3 for duration, _prompt in holds)
    assert total_seconds(steps) == 12
    prompts = " ".join(prompt for _duration, prompt in steps)
    for pose in ("右手", "左手", "掌心", "手背"):
        assert pose in prompts


def _steps(n: int = 6, dur: float = 5.0) -> list[tuple[float, str]]:
    return [(dur, f"seg{i}") for i in range(1, n + 1)]


def _feed(
    controller: GateDiagController,
    start: float,
    frames: list[tuple[int, float, float]],
    dt: float = 1.0 / 30.0,
) -> tuple[list, float]:
    """喂帧序列 (hand_count, palm_width, std)，返回 (事件列表, 结束时刻)。"""

    events = []
    for index, frame in enumerate(frames):
        events.extend(controller.observe(start + index * dt, *frame))
    return events, start + len(frames) * dt


GOOD = (1, 0.20, 40.0)


def test_precheck_passes_after_exactly_30_consecutive_good_frames():
    gate = GateDiagController(_steps())
    events, _ = _feed(gate, 100.0, [GOOD] * 29)
    assert events == []
    assert gate.phase == "precheck"

    events, _ = _feed(gate, 101.0, [GOOD])
    assert [event.kind for event in events] == ["precheck_pass", "step"]
    assert events[0].line.startswith("[预检通过]")
    assert "(1/6)" in events[1].line
    assert gate.phase == "running"
    assert gate.hud_alert is False
    assert "gate-diag 1/6" in gate.hud_text()


def test_precheck_resets_on_missing_hand_or_small_palm():
    gate = GateDiagController(_steps())

    # 29 帧好画面后手丢一帧 → 计数清零。
    _feed(gate, 0.0, [GOOD] * 29 + [(0, 0.0, 40.0)])
    events, _ = _feed(gate, 2.0, [GOOD] * 29)
    assert events == []  # streak 回到 29，仍不够

    # 第 30 帧 palm_width 卡在阈值下沿（0.149 < 0.15）再次清零，
    # 之后要重新攒满 30 帧才放行。
    events, _ = _feed(gate, 5.0, [(1, 0.149, 40.0)] + [GOOD] * 30)
    assert [event.kind for event in events][:2] == ["precheck_pass", "step"]


def test_precheck_threshold_is_inclusive():
    gate = GateDiagController(_steps())
    events, _ = _feed(gate, 0.0, [(1, 0.15, 40.0)] * 30)
    assert events[0].kind == "precheck_pass"


def test_hud_shows_precheck_progress_and_metrics():
    gate = GateDiagController(_steps())
    _feed(gate, 0.0, [GOOD] * 12)
    text = gate.hud_text()
    assert "预检 12/30" in text
    assert "hand=1" in text
    assert "palm=0.20" in text
    assert gate.hud_alert is True


def test_seg1_early_abort_on_gray_wall():
    gate = GateDiagController(_steps())
    _feed(gate, 0.0, [GOOD] * 30)  # 预检通过，seg1 开始

    # 29 帧无手+无纹理尚不足连续 30 帧（紧接预检帧，elapsed 仍 <3s）。
    events, _ = _feed(gate, 1.0, [(0, 0.0, 15.0)] * 29)
    assert events == []
    assert gate.phase == "running"

    events, _ = _feed(gate, 2.0, [(0, 0.0, 15.0)])
    assert len(events) == 1
    assert events[0].kind == "abort"
    assert "std<20" in events[0].line
    assert gate.phase == "aborted"
    assert gate.hud_alert is True


def test_seg1_abort_when_hand_never_seen_even_if_textured():
    gate = GateDiagController(_steps())
    _feed(gate, 0.0, [GOOD] * 30)

    # std 高（有纹理的空房间）但前 3s 始终无手：不触发早退，到 3s 判定中止。
    events, _ = _feed(gate, 1.0, [(0, 0.0, 40.0)] * 95)  # 末帧 elapsed≈3.17s
    kinds = [event.kind for event in events]
    assert kinds == ["abort"]
    assert "hand_count" in events[0].line
    assert gate.phase == "aborted"


def test_seg1_abort_when_low_std_frames_majority():
    gate = GateDiagController(_steps())
    _feed(gate, 0.0, [GOOD] * 30)

    # hand_count=1（偶发误检）但画面近纯色占比 100% → 3s 时按无纹理中止。
    events, _ = _feed(gate, 1.0, [(1, 0.16, 10.0)] * 95)
    kinds = [event.kind for event in events]
    assert kinds == ["abort"]
    assert "std" in events[0].line


def test_seg1_valid_scene_advances_to_step2_at_5s():
    gate = GateDiagController(_steps())
    _feed(gate, 0.0, [GOOD] * 30)

    # 稳定有效画面 5 秒：3s 守卫不中止，5s 边界切 seg2。
    events, _ = _feed(gate, 1.0, [GOOD] * 151)
    assert not any(e.kind == "abort" for e in events)
    step_events = [e for e in events if e.kind == "step"]
    assert len(step_events) == 1
    assert "(2/6)" in step_events[0].line
    assert gate.phase == "running"


def test_full_run_emits_all_steps_and_done():
    gate = GateDiagController(_steps())
    events, _ = _feed(gate, 0.0, [GOOD] * 30)
    assert len(events) == 2  # precheck_pass + seg1
    events2, _ = _feed(gate, 1.0, [GOOD] * 901)  # 30.03s，覆盖全部边界

    step_lines = [e.line for e in events if e.kind == "step"]
    step_lines += [e.line for e in events2 if e.kind == "step"]
    assert len(step_lines) == 6
    for index in range(1, 7):
        assert any(f"({index}/6)" in line for line in step_lines)
    assert events2[-1].kind == "done"
    assert gate.phase == "done"


def test_aborted_controller_goes_silent():
    gate = GateDiagController(_steps())
    _feed(gate, 0.0, [GOOD] * 30 + [(0, 0.0, 15.0)] * 30)
    assert gate.phase == "aborted"

    events, _ = _feed(gate, 20.0, [GOOD] * 100)
    assert events == []  # 中止后再喂好画面也不复活、不推进
