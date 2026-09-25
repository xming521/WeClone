import pytest

from weclone.utils.retry import _calculate_delay


@pytest.mark.parametrize(
    "attempt,jitter_fraction,expected",
    [(0, 1, 1.2), (6, 1, 60.0), (6, -1, 48.0), (6, 0, 60.0)],
)
def test_jitter_respects_max_delay(monkeypatch, attempt, jitter_fraction, expected):
    monkeypatch.setattr(
        "weclone.utils.retry.random.uniform", lambda low, high: high * jitter_fraction
    )

    assert _calculate_delay(attempt, 1.0, 60.0, 2.0, True) == pytest.approx(expected)


def test_delay_without_jitter_respects_max_delay():
    assert _calculate_delay(6, 1.0, 60.0, 2.0, False) == 60.0
