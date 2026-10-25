"""首次使用向导的纯逻辑测试（交互部分在 opencv 窗口里，无法 headless 验证）。"""

import pytest

from handmouse.collect import PRACTICE_STEPS, STEPS
from handmouse.config import AppConfig
from handmouse.hotkeys import HotkeyMonitor
from handmouse.onboarding import (
    CALIBRATION_TARGETS,
    SpeedProfile,
    margins_from_comfort,
    margins_from_points,
    speed_profile,
)


def test_comfort_point_near_center_keeps_defaults() -> None:
    """舒适点就在正中附近时不改配置：一次偶然的偏斜不该写进活动范围。"""

    assert margins_from_comfort(0.5, 0.5) is None
    assert margins_from_comfort(0.53, 0.54) is None


def test_margins_center_on_comfort_point() -> None:
    """活动区的中心必须落在舒适点上——这样抬手够顶和够底的距离一样。"""

    cx, cy = 0.62, 0.45
    left, top, right, bottom = margins_from_comfort(cx, cy)

    assert (left + (1.0 - right)) / 2 == pytest.approx(cx)
    assert (top + (1.0 - bottom)) / 2 == pytest.approx(cy)


def test_margins_approach_comfort_point_when_it_sits_near_the_edge() -> None:
    """舒适点太靠下时中心只能"尽量"贴近它。

    ``pointer_margin_*`` 的合法上限是 0.35（见 `config.validate`），所以
    活动区最靠下也只能是 ``[0.35, 1.0]``、中心 0.675。用户落在 0.72 时，
    只差 0.045（1920×1080 上约 48px 的等效行程），**不能为了对齐它把
    活动区缩成一条**——那会让光标一碰就飞。
    """

    left, top, right, bottom = margins_from_comfort(0.62, 0.72)
    center_y = (top + (1.0 - bottom)) / 2

    assert top == pytest.approx(0.35)  # 已经顶到合法上限
    assert center_y == pytest.approx(0.675)
    assert abs(center_y - 0.72) < 0.05


def test_margins_are_clamped_inside_the_frame() -> None:
    """贴边的舒适点不能算出负内缩（负值等于把画面外也算进活动区）。"""

    for cx, cy in ((0.02, 0.02), (0.98, 0.98), (0.0, 1.0)):
        left, top, right, bottom = margins_from_comfort(cx, cy)
        assert 0.0 <= left <= 0.35
        assert 0.0 <= top <= 0.35
        assert 0.0 <= right <= 0.35
        assert 0.0 <= bottom <= 0.35
        # 活动区不能为空
        assert left + right < 1.0
        assert top + bottom < 1.0


def test_margins_pass_config_validation() -> None:
    """算出来的值必须能过 config.validate()，否则保存时会被悄悄改掉。"""

    left, top, right, bottom = margins_from_comfort(0.75, 0.30)
    config = AppConfig()
    config.pointer_margin_left = left
    config.pointer_margin_top = top
    config.pointer_margin_right = right
    config.pointer_margin_bottom = bottom
    config.validate()

    assert config.pointer_margin_left == pytest.approx(left)
    assert config.pointer_margin_bottom == pytest.approx(bottom)


def _straight_track(speed: float, count: int = 60, dt: float = 1 / 30) -> list[tuple[float, float, float]]:
    """匀速水平移动的轨迹，速度单位 = 画面宽/秒。"""

    return [(index * dt, 0.2 + index * dt * speed, 0.5) for index in range(count)]


def test_speed_profile_measures_constant_speed() -> None:
    profile = speed_profile(_straight_track(0.5))

    assert profile is not None
    assert profile.p50 == pytest.approx(0.5, rel=0.05)
    assert profile.p90 == pytest.approx(0.5, rel=0.05)
    assert profile.label == "中等"


@pytest.mark.parametrize(
    ("speed", "keyword"),
    [
        (0.05, "很稳"),
        (0.20, "偏慢"),
        (0.50, "中等"),
        (1.20, "偏快"),
    ],
)
def test_speed_profile_labels(speed: float, keyword: str) -> None:
    """边界按 p90 划（p50 会被"站着看提示语"的近静止帧拉低）。"""

    profile = speed_profile(_straight_track(speed))

    assert profile is not None
    assert keyword in profile.label


def test_speed_label_uses_p90_not_p50() -> None:
    """真实数据（p50=0.029 / p90=0.090）当初被判成"偏慢"到离谱 —— 现在看 p90。"""

    still = SpeedProfile(p50=0.029, p90=0.090, samples=112)
    moving = SpeedProfile(p50=0.029, p90=0.600, samples=112)

    assert "很稳" in still.label
    assert moving.label == "中等"


def test_speed_profile_needs_enough_samples() -> None:
    """样本太少就不要下结论——宁可返回 None 也别记一个假速度。"""

    assert speed_profile(_straight_track(1.0, count=5)) is None
    assert speed_profile([]) is None


def test_speed_profile_ignores_long_gaps() -> None:
    """停顿/丢帧造成的长间隔不能算进速度（否则会被算成"很慢"）。"""

    track = _straight_track(1.0, count=30)
    track.append((track[-1][0] + 3.0, 0.9, 0.5))  # 停了 3 秒才出现下一帧

    profile = speed_profile(track)

    assert profile is not None
    assert profile.p50 == pytest.approx(1.0, rel=0.05)


def test_speed_profile_label_is_a_plain_string() -> None:
    """label 是给人看的，不能因为数据异常抛异常。"""

    profile = SpeedProfile(p50=0.0, p90=0.0, samples=0)
    assert isinstance(profile.label, str) and profile.label


def test_practice_steps_cover_the_same_gestures_but_shorter() -> None:
    """向导的练习集必须覆盖采集集里的**每一种**手势，只是时间更短。"""

    assert [step.label for step in PRACTICE_STEPS] == [step.label for step in STEPS]
    for practice, full in zip(PRACTICE_STEPS, STEPS):
        assert practice.record_seconds < full.record_seconds
        assert practice.cue == full.cue


def test_calibration_targets_are_inside_the_frame() -> None:
    """标定十字必须落在画面里（它只表示屏幕方位，但画到画面外就看不到了）。"""

    assert len(CALIBRATION_TARGETS) >= 3, "用户要求：不能只标一个点"
    for target in CALIBRATION_TARGETS:
        x, y = target.mark
        assert 0.2 < x < 0.8
        assert 0.2 < y < 0.8
        assert target.label and target.hint


def test_calibration_targets_spread_over_the_frame() -> None:
    """三个点要真的分散在画面不同方位，否则量不出范围。"""

    xs = [target.mark[0] for target in CALIBRATION_TARGETS]
    ys = [target.mark[1] for target in CALIBRATION_TARGETS]

    assert max(xs) - min(xs) > 0.2
    assert max(ys) - min(ys) > 0.2


def test_multi_point_range_becomes_activity_area() -> None:
    """三个点圈出的范围要变成活动区，且中心落在三点中间。"""

    # 中心放 0.5、跨度 0.30：算出来的四边内缩都低于 0.35 上限，不会被夹
    points = [(0.35, 0.35), (0.50, 0.50), (0.65, 0.65)]

    left, top, right, bottom = margins_from_points(points)

    cx = (left + (1.0 - right)) / 2
    cy = (top + (1.0 - bottom)) / 2
    assert cx == pytest.approx(0.50, abs=1e-6)
    assert cy == pytest.approx(0.50, abs=1e-6)
    # 范围被放大 1.25 倍：span 0.30 → 半宽 0.1875，宽度 0.375
    assert (1.0 - right) - left == pytest.approx(0.30 * 1.25, rel=1e-6)


def test_points_too_close_fall_back_to_single_point() -> None:
    """三个点都捏在同一处 = 用户没给范围，硬按小范围映射全屏会"一碰就飞"。"""

    tight = [(0.50, 0.50), (0.51, 0.50), (0.50, 0.51)]

    # 与单点路径等价（都贴近正中 → 不改配置）
    assert margins_from_points(tight) == margins_from_comfort(0.50, 0.50)


def test_multi_point_area_respects_the_035_margin_cap() -> None:
    """范围再大也不能突破 pointer_margin 的合法上限（否则保存时被悄悄改掉）。"""

    wide = [(0.05, 0.05), (0.5, 0.5), (0.95, 0.95)]

    left, top, right, bottom = margins_from_points(wide)

    for value in (left, top, right, bottom):
        assert 0.0 <= value <= 0.35


def test_empty_points_change_nothing() -> None:
    assert margins_from_points([]) is None


def test_onboarding_done_defaults_to_false() -> None:
    """新用户默认"没录过"——这是首次提醒的唯一依据。"""

    assert AppConfig().onboarding_done is False

    config = AppConfig()
    config.onboarding_done = "yes"  # type: ignore[assignment]
    config.validate()
    assert config.onboarding_done is True


def test_hotkey_monitor_accepts_preview_callback() -> None:
    """Ctrl+Alt+P 是预览窗的唯一入口（窗关掉之后窗口内按键就没了）。"""

    calls: list[str] = []
    monitor = HotkeyMonitor(
        lambda: calls.append("toggle"),
        lambda: calls.append("emergency"),
        lambda: calls.append("quit"),
        lambda: calls.append("preview"),
    )

    assert monitor.on_preview is not None
    monitor.on_preview()
    assert calls == ["preview"]


def test_hotkey_monitor_preview_is_optional() -> None:
    monitor = HotkeyMonitor(lambda: None, lambda: None, lambda: None)
    assert monitor.on_preview is None


def test_onboarding_appends_instead_of_replacing_existing_samples(tmp_path) -> None:
    """重跑向导不能把已有样本库替换掉（早期实现用 dataset=None 新建，会覆盖整库）。

    这里不跑交互流程，只锁住"从哪个文件读、以什么模式交给采集"这条契约：
    `run_onboarding` 必须在**已有文件存在时**载入它并作为 `dataset` 传入。
    """

    import inspect

    from handmouse import onboarding

    source = inspect.getsource(onboarding.run_onboarding)

    assert "default_dataset_path()" in source
    # 关键断言：把已有样本库作为 dataset 传下去（追加），而不是新建
    assert "dataset=existing" in source
    assert "GestureDataset.load" in source


def test_onboarding_module_is_runnable_with_dash_m() -> None:
    """`python -m handmouse.onboarding` 必须真的能跑。

    向导中断时会打印"随时可以重新跑：python -m handmouse.onboarding" ——
    少了 `if __name__ == "__main__"` 守卫，那条命令会**静默什么都不做**
    （退出码 0、没有任何输出），用户以为跑了，其实没跑。
    """

    import inspect

    from handmouse import collect, onboarding

    for module in (onboarding, collect):
        source = inspect.getsource(module)
        assert '__name__ == "__main__"' in source, module.__name__


def test_degenerate_axis_falls_back_to_default_scale() -> None:
    """某一轴跨度为 0 时必须退回默认尺度，不能套下限。

    真实数据（2026-10-25 作者第一次三点标定）：y = 0.709 / 0.686 / 0.686，
    纵向跨度只有 0.0225。当时按 `_MIN_HALF`(0.12) 兜底 → 活动区只有 0.47 高，
    比默认 0.68 **敏感 37%**，手一抖光标就飞。跨度小 = 没拿到这个方向的范围，
    就该用默认尺度。
    """

    points = [(0.4889, 0.7087), (0.5084, 0.6862), (0.7125, 0.6862)]

    left, top, right, bottom = margins_from_points(points)
    height = (1.0 - bottom) - top

    # 纵向退回默认 0.68 左右（会被 0.35 的上限挤掉一点）
    assert height > 0.60, f"纵向活动区太小，会过度敏感：{height:.3f}"


def test_axis_half_uses_default_when_span_is_tiny() -> None:
    from handmouse.onboarding import _axis_half

    assert _axis_half(0.0, 0.34) == 0.34
    assert _axis_half(0.05, 0.34) == 0.34
    # 有真实跨度时按 span/2×1.25 放大，夹在 [下限, 默认] 之间
    assert _axis_half(0.40, 0.34) == pytest.approx(0.25)
    assert _axis_half(0.20, 0.34) == pytest.approx(0.125)
    # 跨度大到算出超过默认值时，仍以默认为上限
    assert _axis_half(1.0, 0.34) == 0.34


def test_app_icon_is_a_real_multi_size_ico() -> None:
    """应用图标必须是多尺寸 ico —— Windows 会按场景挑 16/32/48/256。"""

    from pathlib import Path

    from PIL import Image

    root = Path(__file__).resolve().parent.parent
    ico = root / "assets" / "hand-solo.ico"
    png = root / "docs" / "icon.png"

    assert ico.exists(), "缺少 assets/hand-solo.ico（打包与任务栏图标都用它）"
    assert png.exists(), "缺少 docs/icon.png（README 顶部展示用）"

    with Image.open(ico) as image:
        sizes = {size for size in image.info.get("sizes", set())}
    # Pillow 读 ico 时 sizes 在 info 里；老版本可能只给当前帧尺寸
    assert (32, 32) in sizes or image.size[0] >= 32
    assert len(sizes) >= 4, f"ico 里尺寸太少：{sorted(sizes)}"

    with Image.open(png) as image:
        assert image.size == (512, 512)
        assert image.mode == "RGBA"


def test_onboarding_reminded_defaults_to_false_and_survives_validate() -> None:
    """提醒只出现一次，靠 onboarding_reminded 记住；默认必须 False。"""

    config = AppConfig()
    assert config.onboarding_reminded is False

    config.onboarding_reminded = 1  # type: ignore[assignment]
    config.validate()
    assert config.onboarding_reminded is True


def test_calibration_only_never_touches_the_sample_library() -> None:
    """`--recalibrate` 只重标定，绝不能碰 samples.jsonl。

    这条用源码契约锁住：重新标定的函数体里**不允许出现 run_collect**。
    用户原话：「采集只采集一次啊，不要每次开启都采集一遍」——换个摄像头
    只想重标定，不该把七种手势再录五分钟。
    """

    import inspect
    import re

    from handmouse import onboarding

    source = inspect.getsource(onboarding.run_calibration_only)
    # 只看"调用"，不看文档里提到的名字
    assert re.search(r"\brun_collect\s*\(", source) is None
    assert "_apply_margins" in source
    assert "ConfigStore().save" in source


def test_frozen_log_dir_follows_project_root(monkeypatch, tmp_path) -> None:
    """打包成 exe 后日志必须写在 exe 所在目录，不能落在解压用的临时目录。

    2026-10-25 打包实测踩到的：`__main__._ensure_output()` 自己拼了
    `Path(__file__).parent.parent / "logs"` —— 单文件 exe 的 `__file__` 在
    PyInstaller 的一次性解压目录里，日志写进去**退出即随目录删除**，
    于是"双击没反应、也找不到日志"。

    🔴 必须在测试函数体内打补丁改 sys.stdout：pytest 的全局捕获会在 fixture
    之后重新赋值，fixture 里改无效。
    """

    import sys

    from handmouse import __main__ as entry

    fake_root = tmp_path / "exe_dir"
    fake_root.mkdir()

    import handmouse.config as config_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", fake_root)

    saved = (sys.stdout, sys.stderr)
    try:
        sys.stdout = None  # type: ignore[assignment]
        sys.stderr = None  # type: ignore[assignment]
        path = entry._ensure_output()
    finally:
        got = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = saved
        for handle in got:
            if handle is not None and hasattr(handle, "close"):
                try:
                    handle.close()
                except Exception:
                    pass

    assert path is not None
    assert path.parent == fake_root / "logs", f"日志跑到了 {path.parent}"


def test_onboarding_hint_adapts_to_frozen_build(monkeypatch) -> None:
    """exe 用户没有 Python，也不该被指去双击 .bat。"""

    import sys

    from handmouse import app as app_module

    # 源码形态
    monkeypatch.delattr(sys, "frozen", raising=False)
    source_hint = " ".join(app_module._onboarding_hint())
    assert "python -m handmouse --onboarding" in source_hint

    # 冻结形态：命令必须是 exe 自己
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"D:/somewhere/HandSolo.exe")
    frozen_hint = " ".join(app_module._onboarding_hint())
    assert "HandSolo.exe --onboarding" in frozen_hint
    assert "python -m handmouse" not in frozen_hint
