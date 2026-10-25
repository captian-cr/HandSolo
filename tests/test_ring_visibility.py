"""状态圆环的显示策略：暂态亮一下就走，正常状态一直在。

对应 2026-10-23 用户的两条要求：
* 「跟着鼠标的那个圈也要跟着鼠标消失就跟着消失」
* 「没有预览窗的话我怕感知不到手指移出了屏幕」
→ 手出画时圆环先变"斜杠"停留 1.2 秒（明确告诉他"是丢手，不是程序崩了"），
  超时后收走，屏幕上不留东西。
"""

from __future__ import annotations

from handmouse.app import _LOST_RING_GRACE_S, _arguments, _ring_visibility


def test_lost_shows_immediately_and_records_clock() -> None:
    show, since = _ring_visibility("lost", None, 100.0)
    assert show is True, "刚丢手时圆环必须还在，否则看不出发生了什么"
    assert since == 100.0


def test_lost_hides_after_grace() -> None:
    _, since = _ring_visibility("lost", None, 100.0)
    show, since = _ring_visibility("lost", since, 100.0 + _LOST_RING_GRACE_S * 0.5)
    assert show is True, "宽限期内不该收走"

    show, since = _ring_visibility(
        "lost", since, 100.0 + _LOST_RING_GRACE_S + 0.01,
    )
    assert show is False, "超过宽限期还留在屏幕上就是碍事"
    assert since == 100.0, "起点不能被刷新，否则宽限期永远重新计时"


def test_paused_also_fades_out() -> None:
    """开机自启默认暂停，屏幕正中不该永久挂一个灰圈。"""

    _, since = _ring_visibility("paused", None, 10.0)
    show, _ = _ring_visibility("paused", since, 10.0 + _LOST_RING_GRACE_S + 1)
    assert show is False


def test_active_gesture_resets_transient_clock() -> None:
    """恢复控制后必须清掉起点，否则下次丢手会"一来就隐藏"。"""

    show, since = _ring_visibility("pointer", 5.0, 20.0)
    assert show is True
    assert since is None

    show, since = _ring_visibility("lost", since, 21.0)
    assert show is True, "刚刚丢手那一帧必须可见"
    assert since == 21.0


def test_no_preview_flag_parses() -> None:
    assert _arguments(["--no-preview"]).no_preview is True
    assert _arguments([]).no_preview is False
