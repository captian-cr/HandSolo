import pytest

from handmouse.calibration import CalibrationError, calculate_calibration


def test_calibration_places_hysteresis_between_closed_and_open_samples() -> None:
    result = calculate_calibration(
        open_ratios=[0.92, 0.96, 1.00, 0.98, 0.94, 0.99, 0.97, 0.95],
        closed_ratios=[0.13, 0.15, 0.14, 0.16, 0.12, 0.15, 0.14, 0.13],
        pointer_points=[
            (0.500, 0.400),
            (0.502, 0.399),
            (0.499, 0.401),
            (0.501, 0.400),
            (0.500, 0.398),
            (0.499, 0.400),
            (0.501, 0.401),
            (0.500, 0.399),
        ],
    )

    assert 0.16 < result.pinch_engage < result.pinch_release < 0.90
    assert result.pinch_release - result.pinch_engage >= 0.06
    assert 0.40 <= result.pointer_stability <= 0.85
    assert 12 <= result.cursor_interpolation_ms <= 60


def test_noisy_pointer_selects_more_stability_than_steady_pointer() -> None:
    steady = [(0.500, 0.400)] * 8
    noisy = [
        (0.490, 0.390),
        (0.510, 0.410),
        (0.488, 0.406),
        (0.512, 0.394),
        (0.493, 0.412),
        (0.507, 0.388),
        (0.489, 0.397),
        (0.511, 0.403),
    ]

    steady_result = calculate_calibration([1.0] * 8, [0.1] * 8, steady)
    noisy_result = calculate_calibration([1.0] * 8, [0.1] * 8, noisy)

    assert noisy_result.pointer_stability > steady_result.pointer_stability
    assert noisy_result.cursor_interpolation_ms >= steady_result.cursor_interpolation_ms


def test_calibration_rejects_too_few_or_indistinguishable_samples() -> None:
    with pytest.raises(CalibrationError, match="样本不足"):
        calculate_calibration([1.0], [0.1], [(0.5, 0.4)])

    with pytest.raises(CalibrationError, match="区分不明显"):
        calculate_calibration(
            [0.50] * 8,
            [0.46] * 8,
            [(0.5, 0.4)] * 8,
        )
