from handmouse.latest_slot import LatestSlot


def test_latest_value_overwrites_backlog() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.put(1)
    slot.put(2)
    slot.put(3)

    value, revision = slot.wait_next(0, timeout=0.01)

    assert value == 3
    assert revision == 3
    assert slot.overwrite_count == 2


def test_wait_timeout_does_not_invent_value() -> None:
    slot: LatestSlot[int] = LatestSlot()
    value, revision = slot.wait_next(0, timeout=0.001)
    assert value is None
    assert revision == 0

