from unittest.mock import Mock

import pytest

from weclone.utils.retry import RetryConfig


@pytest.mark.parametrize("statuses, expected_calls", [(None, 2), ([], 1), ([429], 2), ([503], 1)])
def test_retry_config_respects_status_list(statuses, expected_calls):
    request = Mock(return_value=Mock(status_code=429))
    config = RetryConfig(max_retries=1, base_delay=0, retry_on_status=statuses)

    assert config.apply_to_function(request)() is request.return_value
    assert request.call_count == expected_calls


@pytest.mark.parametrize(
    "exceptions, expected_calls", [(None, 2), ([], 1), ([TimeoutError], 2), ([ConnectionError], 1)]
)
def test_retry_config_respects_exception_list(exceptions, expected_calls):
    request = Mock(side_effect=TimeoutError("request timed out"))
    config = RetryConfig(max_retries=1, base_delay=0, retry_on_exceptions=exceptions)

    with pytest.raises(TimeoutError, match="request timed out"):
        config.apply_to_function(request)()
    assert request.call_count == expected_calls
