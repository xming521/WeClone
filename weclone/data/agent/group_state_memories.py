from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import pyjson5

DEFAULT_INPUT_PATH = Path("dataset/res_csv/agent/people")
DEFAULT_OUTPUT_DIR = Path("dataset/res_csv/agent/memory_grouping")
DEFAULT_CONFIG_PATH = Path("settings.jsonc")
DEFAULT_EMBEDDING_SERVICE_SCRIPT = Path("weclone/core/inference/embedding_service.py")
DEFAULT_EMBEDDING_SERVICE_LOG_PATH = Path("embedding_service.log")
DEFAULT_AUTO_START_EMBEDDING_SERVICE = True
DEFAULT_STOP_STARTED_EMBEDDING_SERVICE = True
DEFAULT_EMBEDDING_SERVICE_READY_TIMEOUT = 300.0
DEFAULT_EMBEDDING_SERVICE_READY_INTERVAL = 5.0
DEFAULT_EMBEDDING_SERVICE_STOP_TIMEOUT = 30.0
DEFAULT_EMBEDDING_BATCH_SIZE = 128
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_MIN_SIMILARITY = 0.83
DEFAULT_TAG_SIMILARITY = 0.78
DEFAULT_NEIGHBOR_TOP_K = 8
DEFAULT_MAX_GROUP_SIZE = 20
DEFAULT_SINGLETON_ABSORB_STRONG_SIMILARITY = DEFAULT_MIN_SIMILARITY
DEFAULT_SINGLETON_ABSORB_WEAK_SIMILARITY = DEFAULT_TAG_SIMILARITY
DEFAULT_SINGLETON_ABSORB_WEAK_TAG_OVERLAP = 0.5
DEFAULT_MAX_ABSORBED_SINGLETONS_PER_GROUP = 12
DEFAULT_LIMIT = None
DEFAULT_INDENT = 2
REPO_ROOT = Path(__file__).resolve().parents[3]

VAGUE_ANCHOR_TERMS = (
    "这个",
    "那个",
    "某个",
    "某件事",
    "相关内容",
    "相关事项",
    "当前任务",
    "相关问题",
    "相关项目",
    "相关材料",
)


@dataclass
class MemoryRecord:
    memory_id: str
    source_index: int
    sample_id: str
    sample_time: str
    chat_with: str
    memory_index: int
    type: str
    content: str
    tags: list[str]
    preference_type: str
    status: str
    importance: int
    confidence: int
    prefilter_status: str
    prefilter_reasons: list[str]
    embedding_text: str


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
        input_path=DEFAULT_INPUT_PATH,
        output_path=None,
        output_dir=DEFAULT_OUTPUT_DIR,
        config_path=DEFAULT_CONFIG_PATH,
        embedding_url=None,
        embedding_service_script=DEFAULT_EMBEDDING_SERVICE_SCRIPT,
        embedding_service_log_path=DEFAULT_EMBEDDING_SERVICE_LOG_PATH,
        auto_start_embedding_service=DEFAULT_AUTO_START_EMBEDDING_SERVICE,
        stop_started_embedding_service=DEFAULT_STOP_STARTED_EMBEDDING_SERVICE,
        embedding_service_ready_timeout=DEFAULT_EMBEDDING_SERVICE_READY_TIMEOUT,
        embedding_service_ready_interval=DEFAULT_EMBEDDING_SERVICE_READY_INTERVAL,
        embedding_service_stop_timeout=DEFAULT_EMBEDDING_SERVICE_STOP_TIMEOUT,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        min_similarity=DEFAULT_MIN_SIMILARITY,
        tag_similarity=DEFAULT_TAG_SIMILARITY,
        neighbor_top_k=DEFAULT_NEIGHBOR_TOP_K,
        max_group_size=DEFAULT_MAX_GROUP_SIZE,
        singleton_absorb_strong_similarity=DEFAULT_SINGLETON_ABSORB_STRONG_SIMILARITY,
        singleton_absorb_weak_similarity=DEFAULT_SINGLETON_ABSORB_WEAK_SIMILARITY,
        singleton_absorb_weak_tag_overlap=DEFAULT_SINGLETON_ABSORB_WEAK_TAG_OVERLAP,
        max_absorbed_singletons_per_group=DEFAULT_MAX_ABSORBED_SINGLETONS_PER_GROUP,
        limit=DEFAULT_LIMIT,
        indent=DEFAULT_INDENT,
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_content_for_key(content: str) -> str:
    return "".join(normalize_text(content).split()).rstrip("。.!！?")


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def unique_texts(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def input_paths_for(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(path for path in input_path.glob("*.json") if path.is_file())
        if paths:
            return paths
        raise FileNotFoundError(f"No JSON files found in input directory: {input_path}")
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def atomic_save_json(path: Path, payload: Any, *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, default=str) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def load_embedding_service_args(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    config = pyjson5.loads(config_path.read_text(encoding="utf-8"))
    args = config.get("embedding_service_args", {})
    return args if isinstance(args, dict) else {}


def service_base_url(args: SimpleNamespace) -> str:
    if args.embedding_url:
        url = normalize_text(args.embedding_url).rstrip("/")
        return url[:-6] if url.endswith("/embed") else url

    config = load_embedding_service_args(args.config_path)
    host = normalize_text(os.environ.get("EMBEDDING_SERVICE_HOST") or config.get("host") or "127.0.0.1")
    port = normalize_text(os.environ.get("EMBEDDING_SERVICE_PORT") or config.get("port") or "8097")
    return f"http://{host}:{port}"


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding service request failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding service request failed: {exc}") from exc


def get_json(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding service health check failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding service health check failed: {exc}") from exc


def resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded

    cwd_path = (Path.cwd() / expanded).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / expanded).resolve()


def is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return host in {"", "127.0.0.1", "localhost", "::1", "0.0.0.0"}


def configure_embedding_service_env(args: SimpleNamespace, base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    config = load_embedding_service_args(Path(args.config_path))
    parsed = urlparse(base_url)

    if parsed.hostname:
        env["EMBEDDING_SERVICE_HOST"] = parsed.hostname
    if parsed.port:
        env["EMBEDDING_SERVICE_PORT"] = str(parsed.port)

    defaults = {
        "EMBEDDING_SERVICE_MODEL": config.get("model_name_or_path"),
        "EMBEDDING_SERVICE_DEVICE": config.get("device"),
        "EMBEDDING_SERVICE_MAX_LENGTH": config.get("max_length"),
        "EMBEDDING_SERVICE_MIN_RETRY_MAX_LENGTH": config.get("min_retry_max_length"),
        "WECLONE_EMBEDDING_REQUEST_TIMEOUT": config.get("request_timeout"),
    }
    for key, value in defaults.items():
        text = normalize_text(value)
        if text and key not in env:
            env[key] = text
    return env


def tail_text(path: Path, *, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def start_embedding_service(args: SimpleNamespace, base_url: str) -> subprocess.Popen[bytes]:
    script_path = resolve_repo_path(Path(args.embedding_service_script))
    if not script_path.is_file():
        raise FileNotFoundError(f"Embedding service script not found: {script_path}")

    log_path = Path(args.embedding_service_log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = configure_embedding_service_env(args, base_url)
    command = (
        [sys.executable, "-m", "weclone.core.inference.embedding_service"]
        if script_path == resolve_repo_path(DEFAULT_EMBEDDING_SERVICE_SCRIPT)
        else [sys.executable, str(script_path)]
    )

    with log_path.open("ab") as log_file:
        marker = (
            f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] "
            f"Starting embedding service: {' '.join(command)}\n"
        )
        log_file.write(marker.encode("utf-8"))
        log_file.flush()
        return subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def stop_embedding_service_process(process: subprocess.Popen[bytes], *, timeout: float) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_for_embedding_service(
    process: subprocess.Popen[bytes],
    *,
    base_url: str,
    request_timeout: float,
    ready_timeout: float,
    ready_interval: float,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + ready_timeout
    last_error: Exception | None = None
    health_url = f"{base_url.rstrip('/')}/health"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            recent_log = tail_text(log_path)
            raise RuntimeError(
                "Embedding service exited before becoming ready. "
                f"log_path={log_path}\n{recent_log}"
            )

        try:
            return get_json(health_url, timeout=min(request_timeout, ready_interval))
        except Exception as exc:
            last_error = exc
            time.sleep(ready_interval)

    recent_log = tail_text(log_path)
    raise TimeoutError(
        f"Embedding service did not become ready within {ready_timeout:.1f}s: {last_error}. "
        f"log_path={log_path}\n{recent_log}"
    )


def ensure_embedding_service(
    args: SimpleNamespace,
    *,
    base_url: str,
) -> tuple[dict[str, Any], subprocess.Popen[bytes] | None]:
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        return get_json(health_url, timeout=args.request_timeout), None
    except Exception as health_error:
        if not args.auto_start_embedding_service:
            raise RuntimeError(
                f"Embedding service is not reachable at {health_url}. "
                "Start it manually or enable auto start."
            ) from health_error

        if not is_local_base_url(base_url):
            raise RuntimeError(
                f"Embedding service is not reachable at {health_url}, and auto start only supports local URLs."
            ) from health_error

        log_path = Path(args.embedding_service_log_path).expanduser()
        print(f"Embedding service 未就绪，正在启动: {base_url} log={log_path}", flush=True)
        process = start_embedding_service(args, base_url)
        try:
            health = wait_for_embedding_service(
                process,
                base_url=base_url,
                request_timeout=args.request_timeout,
                ready_timeout=args.embedding_service_ready_timeout,
                ready_interval=args.embedding_service_ready_interval,
                log_path=log_path,
            )
        except Exception:
            stop_embedding_service_process(
                process,
                timeout=args.embedding_service_stop_timeout,
            )
            raise
        print(f"Embedding service 已就绪: {base_url}", flush=True)
        return health, process


def embed_texts(
    texts: Sequence[str],
    *,
    base_url: str,
    batch_size: int,
    timeout: float,
) -> list[list[float]]:
    if not texts:
        return []
    payload = post_json(
        f"{base_url.rstrip('/')}/embed",
        {"texts": list(texts), "batch_size": int(batch_size)},
        timeout=timeout,
    )
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError("Embedding service returned invalid payload: missing embeddings list.")
    return embeddings


def raw_tag_set(record: MemoryRecord) -> set[str]:
    return {normalize_text(tag) for tag in record.tags if normalize_text(tag)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def has_vague_anchor(content: str) -> bool:
    return any(term in content for term in VAGUE_ANCHOR_TERMS)


def prefilter_memory(
    *,
    content: str,
    memory_type: str,
    tags: list[str],
    importance: int,
    confidence: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not content:
        return "discard", ["empty_content"]

    if not content.startswith("B"):
        reasons.append("owner_not_explicit")

    if not tags:
        reasons.append("no_tags")
    if has_vague_anchor(content):
        reasons.append("vague_anchor")
    if confidence <= 2:
        reasons.append("low_confidence")
    if importance <= 1:
        reasons.append("low_importance")
    if memory_type not in {"stable_fact", "preference", "goal", "current_state"}:
        reasons.append("unknown_type")

    retrievable = "vague_anchor" not in reasons
    useful = importance >= 2 or confidence >= 3
    if "unknown_type" in reasons:
        return "archive", reasons
    if not retrievable:
        return "archive", reasons
    if not useful:
        return "archive", reasons
    return "candidate", reasons


def embedding_text_for(
    *,
    content: str,
    memory_type: str,
    tags: list[str],
) -> str:
    pieces = [
        f"type: {memory_type}",
        f"content: {content}",
        f"raw_tags: {'; '.join(tags)}" if tags else "",
    ]
    return "\n".join(piece for piece in pieces if piece)


def sample_id_for(item: dict[str, Any], source_index: int) -> str:
    for key in ("id", "local_id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(source_index)


def extract_memory_records(data: list[Any], *, limit: int | None = None) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for source_index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        sample_id = sample_id_for(item, source_index)
        memories = item.get("state_memories", [])
        if not isinstance(memories, list):
            continue
        for memory_index, memory in enumerate(memories):
            if not isinstance(memory, dict):
                continue
            content = normalize_text(memory.get("content"))
            memory_type = normalize_text(memory.get("type"))
            tags = unique_texts(memory.get("tags") or [])
            importance = coerce_int(memory.get("importance"), 0)
            confidence = coerce_int(memory.get("confidence"), 0)
            preference_type = normalize_text(memory.get("preference_type"))
            status = normalize_text(memory.get("status"))
            prefilter_status, prefilter_reasons = prefilter_memory(
                content=content,
                memory_type=memory_type,
                tags=tags,
                importance=importance,
                confidence=confidence,
            )
            record = MemoryRecord(
                memory_id=f"{sample_id}:{memory_index}",
                source_index=source_index,
                sample_id=sample_id,
                sample_time=normalize_text(item.get("time")),
                chat_with=normalize_text(item.get("chat_with")),
                memory_index=memory_index,
                type=memory_type,
                content=content,
                tags=tags,
                preference_type=preference_type,
                status=status,
                importance=importance,
                confidence=confidence,
                prefilter_status=prefilter_status,
                prefilter_reasons=prefilter_reasons,
                embedding_text=embedding_text_for(
                    content=content,
                    memory_type=memory_type,
                    tags=tags,
                ),
            )
            records.append(record)
            if limit is not None and len(records) >= limit:
                return records
    return records


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    score = sum(float(left[index]) * float(right[index]) for index in range(length))
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score


def should_link_records(
    left: MemoryRecord,
    right: MemoryRecord,
    *,
    similarity: float,
    min_similarity: float,
    tag_similarity: float,
) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    if normalize_content_for_key(left.content) == normalize_content_for_key(right.content):
        return True, ["exact_content_duplicate"], 1.0

    if left.type != right.type:
        return False, [], 0.0

    raw_overlap = jaccard(raw_tag_set(left), raw_tag_set(right))

    if similarity >= min_similarity:
        reasons.append("embedding_similarity")
    if raw_overlap > 0:
        reasons.append("raw_tag_overlap")

    if similarity >= min_similarity:
        return True, reasons, raw_overlap
    if similarity >= tag_similarity and raw_overlap > 0:
        return True, reasons, raw_overlap
    return False, [], raw_overlap


def build_candidate_edges(
    records: list[MemoryRecord],
    embeddings: list[list[float]],
    *,
    min_similarity: float,
    tag_similarity: float,
    neighbor_top_k: int,
) -> list[dict[str, Any]]:
    raw_edges: list[dict[str, Any]] = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            similarity = cosine_similarity(embeddings[left_index], embeddings[right_index])
            should_link, reasons, tag_overlap = should_link_records(
                records[left_index],
                records[right_index],
                similarity=similarity,
                min_similarity=min_similarity,
                tag_similarity=tag_similarity,
            )
            if not should_link:
                continue
            raw_edges.append(
                {
                    "left_id": records[left_index].memory_id,
                    "right_id": records[right_index].memory_id,
                    "similarity": round(similarity, 6),
                    "tag_overlap": round(tag_overlap, 6),
                    "reasons": reasons,
                }
            )

    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in raw_edges:
        by_record[edge["left_id"]].append(edge)
        by_record[edge["right_id"]].append(edge)

    selected_keys: set[tuple[str, str]] = set()
    for memory_id, edges in by_record.items():
        ranked = sorted(
            edges,
            key=lambda edge: (
                edge["similarity"],
                edge["tag_overlap"],
                "exact_content_duplicate" in edge["reasons"],
            ),
            reverse=True,
        )
        for edge in ranked[:neighbor_top_k]:
            key = tuple(sorted((edge["left_id"], edge["right_id"])))
            selected_keys.add(key)

    selected_edges = [
        edge
        for edge in raw_edges
        if tuple(sorted((edge["left_id"], edge["right_id"]))) in selected_keys
    ]
    return sorted(
        selected_edges,
        key=lambda edge: (edge["similarity"], edge["tag_overlap"]),
        reverse=True,
    )


def group_key_for(record: MemoryRecord) -> str:
    detail = normalize_text(record.tags[0] if record.tags else "")
    return " / ".join(part for part in (record.type, detail) if part)


def edge_neighbor_id(edge: dict[str, Any], memory_id: str) -> str:
    return edge["right_id"] if edge["left_id"] == memory_id else edge["left_id"]


def should_absorb_singleton_edge(
    edge: dict[str, Any],
    *,
    strong_similarity: float,
    weak_similarity: float,
    weak_tag_overlap: float,
) -> bool:
    similarity = float(edge.get("similarity") or 0.0)
    tag_overlap = float(edge.get("tag_overlap") or 0.0)
    return similarity >= strong_similarity or (
        similarity >= weak_similarity and tag_overlap >= weak_tag_overlap
    )


def refresh_group_fields(
    group: dict[str, Any],
    *,
    record_by_id: dict[str, MemoryRecord],
) -> None:
    member_ids = [normalize_text(memory_id) for memory_id in group.get("memory_ids", [])]
    members = [record_by_id[memory_id] for memory_id in member_ids if memory_id in record_by_id]
    group_tags = Counter()
    for member in members:
        group_tags.update(raw_tag_set(member))
    supporting_edges = group.get("supporting_edges", [])

    group["size"] = len(member_ids)
    group["top_tags"] = [
        {"tag": tag, "count": count}
        for tag, count in group_tags.most_common(10)
    ]
    group["max_similarity"] = max(
        (edge["similarity"] for edge in supporting_edges),
        default=None,
    )
    group["edge_reasons"] = sorted(
        {
            reason
            for edge in supporting_edges
            for reason in edge.get("reasons", [])
        }
    )
    group["members"] = [asdict(member) for member in members]


def absorb_singletons_into_groups(
    groups: list[dict[str, Any]],
    singletons: list[str],
    edges: list[dict[str, Any]],
    *,
    record_by_id: dict[str, MemoryRecord],
    strong_similarity: float,
    weak_similarity: float,
    weak_tag_overlap: float,
    max_per_group: int,
) -> tuple[list[str], int]:
    if not groups or not singletons or max_per_group <= 0:
        return singletons, 0

    singleton_set = set(singletons)
    groups_by_id = {normalize_text(group.get("group_id")): group for group in groups}
    member_to_group: dict[str, str] = {}
    for group_id, group in groups_by_id.items():
        for memory_id in group.get("memory_ids", []):
            member_to_group[normalize_text(memory_id)] = group_id

    best_by_singleton: dict[str, tuple[str, dict[str, Any], str]] = {}
    for edge in edges:
        left_id = normalize_text(edge.get("left_id"))
        right_id = normalize_text(edge.get("right_id"))
        left_singleton = left_id in singleton_set
        right_singleton = right_id in singleton_set
        if left_singleton == right_singleton:
            continue

        singleton_id = left_id if left_singleton else right_id
        neighbor_id = right_id if left_singleton else left_id
        group_id = member_to_group.get(neighbor_id)
        if not group_id:
            continue
        if not should_absorb_singleton_edge(
            edge,
            strong_similarity=strong_similarity,
            weak_similarity=weak_similarity,
            weak_tag_overlap=weak_tag_overlap,
        ):
            continue

        next_score = (
            float(edge.get("similarity") or 0.0),
            float(edge.get("tag_overlap") or 0.0),
        )
        current = best_by_singleton.get(singleton_id)
        current_score = (
            float(current[1].get("similarity") or 0.0),
            float(current[1].get("tag_overlap") or 0.0),
        ) if current else (-1.0, -1.0)
        if next_score > current_score:
            best_by_singleton[singleton_id] = (group_id, edge, neighbor_id)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for singleton_id, (group_id, edge, neighbor_id) in best_by_singleton.items():
        by_group[group_id].append(
            {
                "singleton_id": singleton_id,
                "edge": edge,
                "neighbor_id": neighbor_id,
                "similarity": float(edge.get("similarity") or 0.0),
                "tag_overlap": float(edge.get("tag_overlap") or 0.0),
            }
        )

    absorbed: set[str] = set()
    touched_group_ids: set[str] = set()
    for group_id, rows in by_group.items():
        group = groups_by_id.get(group_id)
        if not group:
            continue
        ranked_rows = sorted(
            rows,
            key=lambda row: (row["similarity"], row["tag_overlap"]),
            reverse=True,
        )
        for row in ranked_rows[:max_per_group]:
            singleton_id = row["singleton_id"]
            if singleton_id in absorbed or singleton_id not in record_by_id:
                continue
            group.setdefault("memory_ids", []).append(singleton_id)
            group.setdefault("supporting_edges", []).append(
                {
                    **row["edge"],
                    "absorbed_singleton_id": singleton_id,
                    "absorbed_via_neighbor_id": row["neighbor_id"],
                }
            )
            group.setdefault("absorbed_singletons", []).append(
                {
                    "memory_id": singleton_id,
                    "neighbor_id": row["neighbor_id"],
                    "similarity": row["edge"].get("similarity"),
                    "tag_overlap": row["edge"].get("tag_overlap"),
                    "reasons": row["edge"].get("reasons", []),
                }
            )
            absorbed.add(singleton_id)
            touched_group_ids.add(group_id)

    for group_id in touched_group_ids:
        refresh_group_fields(groups_by_id[group_id], record_by_id=record_by_id)

    return [memory_id for memory_id in singletons if memory_id not in absorbed], len(absorbed)


def build_candidate_groups(
    records: list[MemoryRecord],
    edges: list[dict[str, Any]],
    *,
    max_group_size: int,
    singleton_absorb_strong_similarity: float,
    singleton_absorb_weak_similarity: float,
    singleton_absorb_weak_tag_overlap: float,
    max_absorbed_singletons_per_group: int,
) -> tuple[list[dict[str, Any]], list[str], int]:
    record_by_id = {record.memory_id: record for record in records}
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["left_id"]].append(edge)
        adjacency[edge["right_id"]].append(edge)

    seed_ids = sorted(
        record_by_id,
        key=lambda memory_id: (
            len(adjacency.get(memory_id, [])),
            record_by_id[memory_id].importance,
            record_by_id[memory_id].confidence,
        ),
        reverse=True,
    )
    assigned: set[str] = set()
    groups: list[dict[str, Any]] = []
    singletons: list[str] = []

    for seed_id in seed_ids:
        if seed_id in assigned:
            continue
        seed_edges = sorted(
            adjacency.get(seed_id, []),
            key=lambda edge: (edge["similarity"], edge["tag_overlap"]),
            reverse=True,
        )
        member_ids = [seed_id]
        supporting_edges: list[dict[str, Any]] = []
        for edge in seed_edges:
            neighbor_id = edge_neighbor_id(edge, seed_id)
            if neighbor_id in assigned or neighbor_id in member_ids:
                continue
            member_ids.append(neighbor_id)
            supporting_edges.append(edge)
            if len(member_ids) >= max_group_size:
                break

        assigned.update(member_ids)
        if len(member_ids) == 1:
            singletons.append(seed_id)
            continue

        members = [record_by_id[memory_id] for memory_id in member_ids]
        group_tags = Counter()
        for member in members:
            group_tags.update(raw_tag_set(member))
        groups.append(
            {
                "group_id": f"group_{len(groups) + 1:04d}",
                "group_key": group_key_for(members[0]),
                "size": len(member_ids),
                "memory_ids": member_ids,
                "top_tags": [
                    {"tag": tag, "count": count}
                    for tag, count in group_tags.most_common(10)
                ],
                "max_similarity": max((edge["similarity"] for edge in supporting_edges), default=None),
                "edge_reasons": sorted(
                    {
                        reason
                        for edge in supporting_edges
                        for reason in edge.get("reasons", [])
                    }
                ),
                "members": [asdict(member) for member in members],
                "supporting_edges": supporting_edges,
            }
        )

    singletons, absorbed_singletons = absorb_singletons_into_groups(
        groups,
        singletons,
        edges,
        record_by_id=record_by_id,
        strong_similarity=singleton_absorb_strong_similarity,
        weak_similarity=singleton_absorb_weak_similarity,
        weak_tag_overlap=singleton_absorb_weak_tag_overlap,
        max_per_group=max_absorbed_singletons_per_group,
    )
    return groups, singletons, absorbed_singletons


def summarize_records(records: list[MemoryRecord]) -> dict[str, Any]:
    by_status = Counter(record.prefilter_status for record in records)
    by_type = Counter(record.type or "unknown" for record in records)
    by_tag = Counter(tag for record in records for tag in raw_tag_set(record))
    reason_counts = Counter(reason for record in records for reason in record.prefilter_reasons)
    return {
        "total": len(records),
        "by_prefilter_status": dict(sorted(by_status.items())),
        "by_type": dict(sorted(by_type.items())),
        "top_tags": [
            {"tag": tag, "count": count}
            for tag, count in by_tag.most_common(20)
        ],
        "prefilter_reason_counts": dict(reason_counts.most_common()),
    }


def output_path_for(args: SimpleNamespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    input_path = Path(args.input_path)
    return Path(args.output_dir) / f"{input_path.stem}.prefilter_candidates.json"


def run(
    args: SimpleNamespace,
    *,
    base_url: str | None = None,
    health: dict[str, Any] | None = None,
    started_embedding_service: bool = False,
) -> dict[str, Any]:
    input_path = Path(args.input_path)
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {input_path}")

    records = extract_memory_records(data, limit=args.limit)
    candidate_records = [record for record in records if record.prefilter_status == "candidate"]
    archive_records = [record for record in records if record.prefilter_status == "archive"]
    discard_records = [record for record in records if record.prefilter_status == "discard"]

    base_url = base_url or service_base_url(args)
    health = health if health is not None else get_json(
        f"{base_url.rstrip('/')}/health",
        timeout=args.request_timeout,
    )
    embeddings = embed_texts(
        [record.embedding_text for record in candidate_records],
        base_url=base_url,
        batch_size=args.embedding_batch_size,
        timeout=args.request_timeout,
    )
    if len(embeddings) != len(candidate_records):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(candidate_records)}, got {len(embeddings)}"
        )

    edges = build_candidate_edges(
        candidate_records,
        embeddings,
        min_similarity=args.min_similarity,
        tag_similarity=args.tag_similarity,
        neighbor_top_k=args.neighbor_top_k,
    )
    groups, singletons, absorbed_singletons = build_candidate_groups(
        candidate_records,
        edges,
        max_group_size=args.max_group_size,
        singleton_absorb_strong_similarity=args.singleton_absorb_strong_similarity,
        singleton_absorb_weak_similarity=args.singleton_absorb_weak_similarity,
        singleton_absorb_weak_tag_overlap=args.singleton_absorb_weak_tag_overlap,
        max_absorbed_singletons_per_group=args.max_absorbed_singletons_per_group,
    )

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_path": str(input_path),
        "embedding_service": {
            "base_url": base_url,
            "health": health,
            "started_by_script": started_embedding_service,
            "batch_size": args.embedding_batch_size,
        },
        "settings": {
            "limit": args.limit,
            "min_similarity": args.min_similarity,
            "tag_similarity": args.tag_similarity,
            "neighbor_top_k": args.neighbor_top_k,
            "max_group_size": args.max_group_size,
            "singleton_absorb_strong_similarity": args.singleton_absorb_strong_similarity,
            "singleton_absorb_weak_similarity": args.singleton_absorb_weak_similarity,
            "singleton_absorb_weak_tag_overlap": args.singleton_absorb_weak_tag_overlap,
            "max_absorbed_singletons_per_group": args.max_absorbed_singletons_per_group,
        },
        "stats": {
            **summarize_records(records),
            "candidate_records": len(candidate_records),
            "archive_records": len(archive_records),
            "discard_records": len(discard_records),
            "candidate_edges": len(edges),
            "candidate_groups": len(groups),
            "absorbed_singletons": absorbed_singletons,
            "singletons": len(singletons),
        },
        "candidate_groups": groups,
        "singletons": singletons,
        "candidate_edges": edges,
        "records": [asdict(record) for record in records],
        "archive_records": [asdict(record) for record in archive_records],
        "discard_records": [asdict(record) for record in discard_records],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prefilter state_memories and build embedding-backed merge candidate groups."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input people JSON file, or a directory containing people JSON files.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--embedding-url", default=None, help="Embedding service base URL or /embed URL.")
    parser.add_argument(
        "--embedding-service-script",
        type=Path,
        default=DEFAULT_EMBEDDING_SERVICE_SCRIPT,
        help="Local embedding service script to start when /health is not reachable.",
    )
    parser.add_argument(
        "--embedding-service-log-path",
        type=Path,
        default=DEFAULT_EMBEDDING_SERVICE_LOG_PATH,
        help="Log file for an embedding service process started by this script.",
    )
    parser.add_argument(
        "--embedding-service-ready-timeout",
        type=float,
        default=DEFAULT_EMBEDDING_SERVICE_READY_TIMEOUT,
        help="Seconds to wait for an auto-started embedding service to become ready.",
    )
    parser.add_argument(
        "--embedding-service-ready-interval",
        type=float,
        default=DEFAULT_EMBEDDING_SERVICE_READY_INTERVAL,
        help="Seconds between embedding service health checks while waiting.",
    )
    parser.add_argument(
        "--embedding-service-stop-timeout",
        type=float,
        default=DEFAULT_EMBEDDING_SERVICE_STOP_TIMEOUT,
        help="Seconds to wait for graceful shutdown before killing an auto-started service.",
    )
    parser.add_argument(
        "--no-auto-start-embedding-service",
        dest="auto_start_embedding_service",
        action="store_false",
        help="Do not start a local embedding service when /health is not reachable.",
    )
    parser.add_argument(
        "--keep-started-embedding-service",
        dest="stop_started_embedding_service",
        action="store_false",
        help="Keep the embedding service running if this script started it.",
    )
    parser.set_defaults(
        auto_start_embedding_service=DEFAULT_AUTO_START_EMBEDDING_SERVICE,
        stop_started_embedding_service=DEFAULT_STOP_STARTED_EMBEDDING_SERVICE,
    )
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--tag-similarity", type=float, default=DEFAULT_TAG_SIMILARITY)
    parser.add_argument("--neighbor-top-k", type=int, default=DEFAULT_NEIGHBOR_TOP_K)
    parser.add_argument("--max-group-size", type=int, default=DEFAULT_MAX_GROUP_SIZE)
    parser.add_argument(
        "--singleton-absorb-strong-similarity",
        type=float,
        default=DEFAULT_SINGLETON_ABSORB_STRONG_SIMILARITY,
    )
    parser.add_argument(
        "--singleton-absorb-weak-similarity",
        type=float,
        default=DEFAULT_SINGLETON_ABSORB_WEAK_SIMILARITY,
    )
    parser.add_argument(
        "--singleton-absorb-weak-tag-overlap",
        type=float,
        default=DEFAULT_SINGLETON_ABSORB_WEAK_TAG_OVERLAP,
    )
    parser.add_argument(
        "--max-absorbed-singletons-per-group",
        type=int,
        default=DEFAULT_MAX_ABSORBED_SINGLETONS_PER_GROUP,
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--indent", type=int, default=DEFAULT_INDENT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    requested_input_path = Path(args.input_path)
    if requested_input_path.is_dir() and args.output_path:
        raise ValueError("--output-path can only be used when --input-path points to one JSON file.")

    input_paths = input_paths_for(requested_input_path)
    base_url = service_base_url(args)
    health, embedding_process = ensure_embedding_service(args, base_url=base_url)

    try:
        for input_path in input_paths:
            run_args = SimpleNamespace(**vars(args))
            run_args.input_path = input_path
            payload = run(
                run_args,
                base_url=base_url,
                health=health,
                started_embedding_service=embedding_process is not None,
            )
            output_path = output_path_for(run_args)
            atomic_save_json(output_path, payload, indent=args.indent)
            stats = payload["stats"]
            print(
                "完成: "
                f"input={input_path} output={output_path} "
                f"records={stats['total']} candidates={stats['candidate_records']} "
                f"archive={stats['archive_records']} discard={stats['discard_records']} "
                f"groups={stats['candidate_groups']} edges={stats['candidate_edges']} "
                f"absorbed_singletons={stats['absorbed_singletons']} "
                f"singletons={stats['singletons']}",
                flush=True,
            )

        if len(input_paths) > 1:
            print(f"全部完成: files={len(input_paths)} output_dir={args.output_dir}", flush=True)
    finally:
        if embedding_process is not None and args.stop_started_embedding_service:
            stop_embedding_service_process(
                embedding_process,
                timeout=args.embedding_service_stop_timeout,
            )
            print("Embedding service 已关闭。", flush=True)


if __name__ == "__main__":
    main()
