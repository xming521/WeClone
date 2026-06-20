from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_REPO_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from weclone.utils.config import load_base_config
    from weclone.utils.log import logger
except ImportError as exc:
    raise RuntimeError(
        "Missing embedding service dependencies. Activate the WeClone environment first."
    ) from exc

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8097
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MIN_RETRY_MAX_LENGTH = 512
DEFAULT_EMBEDDER_MAX_LENGTH = 8192
DEFAULT_MODEL = "./models/Qwen3-Embedding-4B"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 128

_embedder: Optional["HFTextEmbedder"] = None
_embedder_lock = Lock()
_service_config: Optional["EmbeddingServiceConfig"] = None


@dataclass(frozen=True)
class EmbeddingServiceConfig:
    model_name_or_path: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_length: int = DEFAULT_MAX_LENGTH
    min_retry_max_length: int = DEFAULT_MIN_RETRY_MAX_LENGTH
    request_timeout: float = DEFAULT_TIMEOUT


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_chunks(items: Sequence[Any], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def resolve_runtime_device(requested_device: Optional[str]) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_model_source(model_path: str | Path) -> str:
    candidate = Path(model_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if candidate.is_absolute():
        try:
            relative_to_legacy_parent = candidate.relative_to(LEGACY_REPO_PARENT)
        except ValueError:
            return str(model_path)

        migrated_candidate = REPO_ROOT / relative_to_legacy_parent
        if migrated_candidate.exists():
            return str(migrated_candidate.resolve())
        return str(model_path)

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    return str(model_path)


class HFTextEmbedder:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = DEFAULT_EMBEDDER_MAX_LENGTH,
        min_retry_max_length: int = DEFAULT_MIN_RETRY_MAX_LENGTH,
    ):
        self.torch = torch
        self.F = F
        self.device = resolve_runtime_device(device)
        self.max_length = max(1, int(max_length))
        self.min_retry_max_length = max(1, int(min_retry_max_length))
        resolved_model_path = resolve_model_source(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModel.from_pretrained(resolved_model_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _normalize_embeddings(self, embeddings: Any) -> list[list[float]]:
        tensor = embeddings
        if not isinstance(tensor, self.torch.Tensor):
            tensor = self.torch.tensor(tensor)
        tensor = tensor.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = self.F.normalize(tensor, p=2, dim=1)
        return tensor.cpu().tolist()

    def _try_model_encode(self, texts: Sequence[str]) -> Optional[list[list[float]]]:
        if not hasattr(self.model, "encode"):
            return None

        with self.torch.inference_mode():
            try:
                embeddings = self.model.encode(list(texts))
            except self.torch.OutOfMemoryError:
                raise
            except TypeError:
                return None
            except Exception:
                return None

        return self._normalize_embeddings(embeddings)

    def _extract_last_hidden_state(self, model_output: Any) -> Any:
        last_hidden_state = getattr(model_output, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state
        if isinstance(model_output, (tuple, list)) and model_output:
            return model_output[0]
        raise RuntimeError("Unable to extract hidden states from embedding model output.")

    def _last_token_pool(self, model_output: Any, attention_mask: Any) -> Any:
        last_hidden_state = self._extract_last_hidden_state(model_output)
        is_left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if is_left_padded:
            return last_hidden_state[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        batch_indices = self.torch.arange(batch_size, device=last_hidden_state.device)
        return last_hidden_state[batch_indices, sequence_lengths]

    def _clear_cuda_cache(self) -> None:
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _encode_batch_once(self, batch: Sequence[str], max_length: int) -> list[list[float]]:
        encoded = self._try_model_encode(batch)
        if encoded is not None:
            return encoded

        with self.torch.inference_mode():
            tokenized = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            outputs = self.model(**tokenized)
            pooled = self._last_token_pool(outputs, tokenized["attention_mask"])
            return self._normalize_embeddings(pooled)

    def _encode_batch_adaptive(self, batch: Sequence[str], max_length: int) -> list[list[float]]:
        try:
            return self._encode_batch_once(batch, max_length=max_length)
        except self.torch.OutOfMemoryError as exc:
            self._clear_cuda_cache()
            current_batch_size = len(batch)

            if current_batch_size > 1:
                split_index = max(1, current_batch_size // 2)
                logger.warning(
                    "Embedding OOM on device={} batch_size={} max_length={}. Retrying with smaller batches.",
                    self.device,
                    current_batch_size,
                    max_length,
                )
                left = self._encode_batch_adaptive(batch[:split_index], max_length=max_length)
                right = self._encode_batch_adaptive(batch[split_index:], max_length=max_length)
                return left + right

            reduced_max_length = max(self.min_retry_max_length, max_length // 2)
            if reduced_max_length < max_length:
                logger.warning(
                    "Embedding OOM on device={} with batch_size=1 max_length={}. Retrying with max_length={}.",
                    self.device,
                    max_length,
                    reduced_max_length,
                )
                return self._encode_batch_adaptive(batch, max_length=reduced_max_length)

            raise RuntimeError(
                "Embedding model ran out of memory even with batch_size=1 "
                f"and max_length={max_length} on device {self.device}."
            ) from exc

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for batch in iter_chunks(list(texts), batch_size):
            embeddings.extend(self._encode_batch_adaptive(batch, max_length=self.max_length))

        return embeddings


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_settings_embedding_args() -> dict[str, Any]:
    try:
        config = load_base_config()
        return config.embedding_service_args.model_dump()
    except SystemExit:
        return {}
    except Exception as exc:
        logger.warning(f"Failed to load embedding_service_args from settings: {exc}")
        return {}


def load_service_config() -> EmbeddingServiceConfig:
    settings = _load_settings_embedding_args()
    model_name_or_path = normalize_text(
        os.environ.get("EMBEDDING_SERVICE_MODEL")
        or settings.get("model_name_or_path")
        or DEFAULT_MODEL
    )
    device = normalize_text(
        os.environ.get("EMBEDDING_SERVICE_DEVICE") or settings.get("device") or DEFAULT_DEVICE
    )
    host = normalize_text(os.environ.get("EMBEDDING_SERVICE_HOST") or settings.get("host") or DEFAULT_HOST)

    return EmbeddingServiceConfig(
        model_name_or_path=model_name_or_path,
        device=device,
        host=host,
        port=_coerce_int(os.environ.get("EMBEDDING_SERVICE_PORT") or settings.get("port"), DEFAULT_PORT),
        max_length=_coerce_int(
            os.environ.get("EMBEDDING_SERVICE_MAX_LENGTH") or settings.get("max_length"),
            DEFAULT_MAX_LENGTH,
        ),
        min_retry_max_length=_coerce_int(
            os.environ.get("EMBEDDING_SERVICE_MIN_RETRY_MAX_LENGTH")
            or settings.get("min_retry_max_length"),
            DEFAULT_MIN_RETRY_MAX_LENGTH,
        ),
        request_timeout=_coerce_float(
            os.environ.get("WECLONE_EMBEDDING_REQUEST_TIMEOUT") or settings.get("request_timeout"),
            DEFAULT_TIMEOUT,
        ),
    )


def get_service_config() -> EmbeddingServiceConfig:
    global _service_config
    if _service_config is None:
        _service_config = load_service_config()
    return _service_config


def get_embedder() -> HFTextEmbedder:
    global _embedder
    config = get_service_config()
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = HFTextEmbedder(
                    config.model_name_or_path,
                    device=config.device,
                    max_length=config.max_length,
                    min_retry_max_length=config.min_retry_max_length,
                )
    return _embedder


def build_health_payload() -> dict[str, str]:
    config = get_service_config()
    return {
        "status": "ok",
        "model": normalize_text(config.model_name_or_path),
        "device": normalize_text(config.device),
        "max_length": str(config.max_length),
    }


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error_response(handler: BaseHTTPRequestHandler, status: int, detail: str) -> None:
    _json_response(handler, status, {"detail": detail})


def _read_json_body(handler: BaseHTTPRequestHandler) -> Any:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if content_length <= 0:
        raise ValueError("request body must not be empty")

    raw_body = handler.rfile.read(content_length)
    try:
        return json.loads(raw_body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("request body must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc


def _parse_embed_payload(payload: Any) -> tuple[list[str], int]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    texts = payload.get("texts", [])
    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("texts must be a list of strings")

    batch_size = payload.get("batch_size", DEFAULT_BATCH_SIZE)
    if isinstance(batch_size, bool):
        raise ValueError("batch_size must be a positive integer")
    try:
        batch_size_int = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size must be a positive integer") from exc
    if batch_size_int <= 0:
        raise ValueError("batch_size must be positive")

    return [normalize_text(text) for text in texts], batch_size_int


class EmbeddingRequestHandler(BaseHTTPRequestHandler):
    server_version = "WeCloneEmbeddingHTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path != "/health":
            _error_response(self, 404, "not found")
            return
        _json_response(self, 200, build_health_payload())

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/embed":
            _error_response(self, 404, "not found")
            return

        try:
            payload = _read_json_body(self)
            texts, batch_size = _parse_embed_payload(payload)
        except ValueError as exc:
            _error_response(self, 400, str(exc))
            return

        if not texts:
            _json_response(self, 200, {"embeddings": []})
            return

        embedder = get_embedder()
        try:
            with _embedder_lock:
                embeddings = embedder.encode_texts(texts, batch_size=batch_size)
        except torch.OutOfMemoryError as exc:
            logger.exception("Embedding request failed with CUDA OOM.")
            _error_response(self, 503, f"CUDA OOM: {exc}")
            return
        except Exception as exc:
            logger.exception("Embedding request failed.")
            _error_response(self, 500, str(exc))
            return

        _json_response(self, 200, {"embeddings": embeddings})


class EmbeddingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global _service_config
    _service_config = load_service_config()

    logger.info(
        "Starting embedding service on "
        "http://"
        f"{_service_config.host}:{_service_config.port} "
        f"using model={_service_config.model_name_or_path} "
        f"device={_service_config.device} "
        f"max_length={_service_config.max_length}"
    )
    get_embedder()

    server = EmbeddingHTTPServer(
        (_service_config.host, _service_config.port),
        EmbeddingRequestHandler,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
