import os
import random
import re
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger as _logger


logger = _logger


PROJECT_ROOT_ENV = "LLM_INFERENCE_PROJECT_ROOT"
OPENROUTER_PROXY_URL_ENV = "WECLONE_OPENROUTER_PROXY_URL"
DEFAULT_OPENROUTER_PROXY_URL = "http://127.0.0.1:7899"
API_TIMEOUT_SECONDS = 45
OPENAI_API_MAX_RETRIES = 100
OPENAI_API_BASE_DELAY = 2.0
OPENAI_API_MAX_DELAY = 60.0
OPENAI_API_BACKOFF_FACTOR = 2.0
OPENAI_API_JITTER = True


def project_root() -> Path:
    configured = os.environ.get(PROJECT_ROOT_ENV)
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def openrouter_proxy_url(base_url: str) -> str | None:
    parsed_url = urlparse(base_url)
    if (parsed_url.hostname or "").lower() != "openrouter.ai":
        return None

    proxy_url = os.environ.get(OPENROUTER_PROXY_URL_ENV, DEFAULT_OPENROUTER_PROXY_URL)
    return proxy_url or None


def calculate_retry_delay(
    attempt: int,
    base_delay: float = OPENAI_API_BASE_DELAY,
    max_delay: float = OPENAI_API_MAX_DELAY,
    backoff_factor: float = OPENAI_API_BACKOFF_FACTOR,
    jitter: bool = OPENAI_API_JITTER,
) -> float:
    delay = min(base_delay * (backoff_factor**attempt), max_delay)
    if jitter:
        jitter_range = delay * 0.2
        delay = max(0, delay + random.uniform(-jitter_range, jitter_range))
    return delay


def extract_json_from_text(text: str) -> str:
    """Extract JSON content from text, including fenced Markdown JSON blocks."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
