import argparse
import copy
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weclone.utils.log import logger

DEFAULT_INPUT_PATH = Path("dataset/res_csv/sft/sft-my.json")
DEFAULT_OUTPUT_DIR = Path("dataset/res_csv/agent/people")
DEFAULT_MANIFEST_NAME = "sft-my-stage2-manifest.json"
DEFAULT_EMPTY_CHAT_WITH = "unknown_chat_with"
DEFAULT_LTP_MODEL_PATH = Path("WC-exp/eval/models/LTPbase")
DEFAULT_LTP_BATCH_SIZE = 64
DEFAULT_LTP_DEVICE = "auto"
DEFAULT_LTP_PHYSICAL_GPU = "1"
BEGIN_CHAT_MARKER = "<begin_chat>"
CONTENT_RATIO_THRESHOLD = 0.3
DEICTIC_RATIO_THRESHOLD = 0.4
FUNCTION_RATIO_THRESHOLD = 0.6
CHAR_LEN_THRESHOLD = 40
DROP_CONDITION_THRESHOLD = 3
VERY_SHORT_CHAR_LEN_THRESHOLD = 25
VERY_SHORT_FUNCTION_RATIO_THRESHOLD = 0.5
HIGH_VALUE_POS = {"n", "nh", "ni", "ns", "nt", "nz", "nl", "nd", "v", "a", "j", "ws", "m"}
LOW_VALUE_POS = {"u", "e", "wp", "r", "o", "c", "p", "d"}
STRONG_REFERENT_POS = {"nh", "ni", "ns", "nt", "nz", "ws"}
ROBUST_CONTENT_POS = {"n", "nl", "nd", "j"}
DEICTIC_KEYWORDS = (
    "这个",
    "那个",
    "这张",
    "那张",
    "这段",
    "那段",
    "这里",
    "那里",
    "这样",
    "那样",
    "图",
    "图片",
    "照片",
    "视频",
    "表情包",
    "表情",
)
LOW_VALUE_KEYWORD_EXCLUSIONS = {
    "知道",
    "觉得",
    "感觉",
    "可以",
    "应该",
    "没有",
    "不是",
    "就是",
    "什么",
    "怎么",
    "这么",
    "那么",
    "一下",
    "一点",
    "一些",
    "一个",
    "几个",
    "多少",
}


def _time_sort_key(item: dict[str, Any], original_index: int):
    value = item.get("time")
    if value is None:
        return (1, "", original_index)

    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return (0, parsed, original_index)
    except ValueError:
        return (1, text, original_index)


def _message_has_begin_chat(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    return BEGIN_CHAT_MARKER in str(message.get("content", ""))


def _remove_system_and_begin_chat_messages(item: dict[str, Any]) -> dict[str, Any]:
    result = {key: copy.deepcopy(value) for key, value in item.items() if key != "system"}
    messages = result.get("messages")
    if isinstance(messages, list):
        result["messages"] = [
            copy.deepcopy(message) for message in messages if not _message_has_begin_chat(message)
        ]
    return result


def _episode_text(item: dict[str, Any]) -> str:
    messages = item.get("messages")
    if not isinstance(messages, list):
        return ""
    contents = []
    for message in messages:
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            if content:
                contents.append(content)
    return "\n".join(contents)


def _char_len_without_spaces(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _high_value_keyword_count(words: list[str], pos_tags: list[str]) -> int:
    count = 0
    for word, pos_tag in zip(words, pos_tags):
        stripped = re.sub(r"\s+", "", str(word))
        if not stripped or stripped in LOW_VALUE_KEYWORD_EXCLUSIONS:
            continue
        if pos_tag in STRONG_REFERENT_POS:
            count += 1
        elif pos_tag in ROBUST_CONTENT_POS and len(stripped) >= 2:
            count += 1
    return count


def _deictic_hit_count(text: str) -> int:
    return sum(text.count(keyword) for keyword in DEICTIC_KEYWORDS)


def _has_useful_anchor(high_value_keywords_count: int) -> bool:
    return high_value_keywords_count >= 2


def _build_density_features(text: str, words: list[str], pos_tags: list[str]) -> dict[str, Any]:
    total_words = len(words)
    high_value_count = sum(1 for pos_tag in pos_tags if pos_tag in HIGH_VALUE_POS)
    low_value_count = sum(1 for pos_tag in pos_tags if pos_tag in LOW_VALUE_POS)
    high_value_keywords_count = _high_value_keyword_count(words, pos_tags)
    deictic_hits = _deictic_hit_count(text)

    if total_words > 0:
        content_ratio = high_value_count / total_words
        function_ratio = low_value_count / total_words
        deictic_ratio = min(deictic_hits / total_words, 1.0)
    else:
        content_ratio = 0.0
        function_ratio = 1.0
        deictic_ratio = 0.0

    char_len = _char_len_without_spaces(text)
    has_useful_anchor = _has_useful_anchor(high_value_keywords_count)
    very_short_low_value = char_len < VERY_SHORT_CHAR_LEN_THRESHOLD and (
        not has_useful_anchor
        and (function_ratio >= VERY_SHORT_FUNCTION_RATIO_THRESHOLD or high_value_keywords_count == 0)
    )
    conditions = {
        "short_char_len": char_len < CHAR_LEN_THRESHOLD,
        "low_content_ratio": content_ratio < CONTENT_RATIO_THRESHOLD,
        "no_high_value_keywords": high_value_keywords_count == 0,
        "high_deictic_ratio": deictic_ratio > DEICTIC_RATIO_THRESHOLD,
        "high_function_ratio": function_ratio > FUNCTION_RATIO_THRESHOLD,
        "very_short_low_value": very_short_low_value,
    }
    matched_conditions = [name for name, matched in conditions.items() if matched]
    base_condition_count = len([name for name in matched_conditions if name != "very_short_low_value"])

    return {
        "char_len": char_len,
        "total_words": total_words,
        "content_ratio": content_ratio,
        "function_ratio": function_ratio,
        "high_value_keywords_count": high_value_keywords_count,
        "has_useful_anchor": has_useful_anchor,
        "deictic_ratio": deictic_ratio,
        "matched_conditions": matched_conditions,
        "drop": base_condition_count >= DROP_CONDITION_THRESHOLD or very_short_low_value,
    }


def _resolve_ltp_map_location(device: str, physical_gpu: str) -> str:
    if device == "cpu":
        return "cpu"
    if device not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported LTP device: {device}")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()
    except Exception:
        cuda_available = False
        cuda_count = 0

    if not cuda_available:
        if device == "cuda":
            raise RuntimeError("LTP device is cuda but CUDA is not available.")
        return "cpu"

    gpu_id = physical_gpu
    if device == "auto":
        try:
            requested_gpu = int(physical_gpu)
        except ValueError:
            requested_gpu = 0
        if requested_gpu >= cuda_count:
            gpu_id = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return "cuda:0"


def _load_ltp(model_path: Path, device: str, physical_gpu: str):
    if not model_path.exists():
        raise FileNotFoundError(f"LTP model path does not exist: {model_path}")

    map_location = _resolve_ltp_map_location(device, physical_gpu)
    from ltp import LTP

    logger.info(f"Loading LTP model from {model_path} on {map_location}")
    return LTP(str(model_path), map_location=map_location)


def _filter_low_information_items(
    items: list[dict[str, Any]],
    ltp,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept_items: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = defaultdict(int)
    filtered_count = 0

    texts = [_episode_text(item) for item in items]
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]
        non_empty_positions = [idx for idx, text in enumerate(batch_texts) if text.strip()]
        ltp_texts = [batch_texts[idx] for idx in non_empty_positions]

        output = ltp.pipeline(ltp_texts, tasks=["cws", "pos"]) if ltp_texts else None  # type: ignore
        ltp_output_index = 0

        for batch_index, item in enumerate(batch_items):
            text = batch_texts[batch_index]
            if batch_index in non_empty_positions:
                words = list(output.cws[ltp_output_index])  # type: ignore[union-attr]
                pos_tags = list(output.pos[ltp_output_index])  # type: ignore[union-attr]
                ltp_output_index += 1
            else:
                words = []
                pos_tags = []

            features = _build_density_features(text, words, pos_tags)
            if features["drop"]:
                filtered_count += 1
                for condition in features["matched_conditions"]:
                    reason_counts[condition] += 1
                continue
            kept_items.append(item)

    return kept_items, {
        "filtered_count": filtered_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _chat_with_filename(chat_with: str, used_filenames: set[str]) -> str:
    name = (chat_with or "").strip() or DEFAULT_EMPTY_CHAT_WITH
    name = name.replace("/", "_").replace("\\", "_")
    if name in {".", ".."}:
        name = DEFAULT_EMPTY_CHAT_WITH

    filename = f"{name}.json"
    if filename not in used_filenames:
        used_filenames.add(filename)
        return filename

    suffix = 2
    while True:
        filename = f"{name}_{suffix}.json"
        if filename not in used_filenames:
            used_filenames.add(filename)
            return filename
        suffix += 1


def _clear_existing_stage2_outputs(output_dir: Path, input_path: Path) -> None:
    if not output_dir.exists():
        return
    if output_dir.resolve() == input_path.parent.resolve():
        raise ValueError(f"Refusing to clear output dir because it is the input file directory: {output_dir}")
    for path in output_dir.glob("*.json"):
        path.unlink()


def _chat_with_id(chat_with_index: int) -> str:
    return f"chat_with_{chat_with_index:04d}"


def _assign_group_ids(items: list[dict[str, Any]], chat_with_id: str) -> list[dict[str, Any]]:
    assigned_items = []
    for local_index, item in enumerate(items):
        assigned_item = copy.deepcopy(item)
        assigned_item["chat_with_id"] = chat_with_id
        assigned_item["local_id"] = str(local_index)
        assigned_items.append(assigned_item)
    return assigned_items


def build_stage2_outputs(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    ltp_model_path: Path | str = DEFAULT_LTP_MODEL_PATH,
    ltp_batch_size: int = DEFAULT_LTP_BATCH_SIZE,
    ltp_device: str = DEFAULT_LTP_DEVICE,
    ltp_physical_gpu: str = DEFAULT_LTP_PHYSICAL_GPU,
) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    manifest_path = output_dir / manifest_name
    ltp_model_path = Path(ltp_model_path)

    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Stage2 input must be a JSON list: {input_path}")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    chat_with_order: list[str] = []
    removed_system_count = 0
    removed_begin_chat_message_count = 0

    for original_index, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict stage2 item at index {original_index}")
            continue

        chat_with = str(item.get("chat_with") or "").strip()
        if chat_with not in grouped:
            chat_with_order.append(chat_with)

        if "system" in item:
            removed_system_count += 1

        messages = item.get("messages")
        if isinstance(messages, list):
            removed_begin_chat_message_count += sum(
                1 for message in messages if _message_has_begin_chat(message)
            )

        grouped[chat_with].append((original_index, item))

    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_stage2_outputs(output_dir, input_path)
    manifest: list[dict[str, Any]] = []
    total_written = 0
    used_filenames: set[str] = set()
    ltp = _load_ltp(ltp_model_path, ltp_device, ltp_physical_gpu)
    total_filtered_low_information_count = 0
    total_reason_counts: dict[str, int] = defaultdict(int)

    for chat_with_index, chat_with in enumerate(chat_with_order):
        items = grouped[chat_with]
        chat_with_id = _chat_with_id(chat_with_index)
        sorted_items = sorted(items, key=lambda pair: _time_sort_key(pair[1], pair[0]))
        processed_items = [_remove_system_and_begin_chat_messages(item) for _, item in sorted_items]
        kept_items, filter_stats = _filter_low_information_items(
            processed_items,
            ltp=ltp,
            batch_size=ltp_batch_size,
        )
        kept_items = _assign_group_ids(kept_items, chat_with_id)
        total_filtered_low_information_count += filter_stats["filtered_count"]
        for reason, count in filter_stats["reason_counts"].items():
            total_reason_counts[reason] += count

        output_filename = _chat_with_filename(chat_with, used_filenames)
        output_path = output_dir / output_filename
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(kept_items, f, ensure_ascii=False, indent=4)

        total_written += len(kept_items)
        manifest.append(
            {
                "chat_with_id": chat_with_id,
                "chat_with": chat_with,
                "file_name": output_filename,
                "output_path": str(output_path),
                "sample_count": len(kept_items),
                "filtered_low_information_count": filter_stats["filtered_count"],
                "filter_reason_counts": filter_stats["reason_counts"],
                "first_time": kept_items[0].get("time") if kept_items else None,
                "last_time": kept_items[-1].get("time") if kept_items else None,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "ltp_model_path": str(ltp_model_path),
                "ltp_batch_size": ltp_batch_size,
                "ltp_device": ltp_device,
                "ltp_physical_gpu": ltp_physical_gpu,
                "removed_system_count": removed_system_count,
                "removed_begin_chat_message_count": removed_begin_chat_message_count,
                "filtered_low_information_count": total_filtered_low_information_count,
                "filter_reason_counts": dict(sorted(total_reason_counts.items())),
                "filter_thresholds": {
                    "char_len_lt": CHAR_LEN_THRESHOLD,
                    "content_ratio_lt": CONTENT_RATIO_THRESHOLD,
                    "high_value_keywords_count_eq": 0,
                    "deictic_ratio_gt": DEICTIC_RATIO_THRESHOLD,
                    "function_ratio_gt": FUNCTION_RATIO_THRESHOLD,
                    "drop_condition_threshold": DROP_CONDITION_THRESHOLD,
                    "very_short_char_len_lt": VERY_SHORT_CHAR_LEN_THRESHOLD,
                    "very_short_function_ratio_gte": VERY_SHORT_FUNCTION_RATIO_THRESHOLD,
                    "very_short_low_value_requires_no_useful_anchor": True,
                },
                "total_written": total_written,
                "groups": manifest,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    logger.success(
        "Stage2 dataset processing successful, "
        f"{len(manifest)} chat_with groups, {total_written} entries, saved to {output_dir}"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build second-stage SFT datasets grouped by chat_with.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--ltp-model-path", type=Path, default=DEFAULT_LTP_MODEL_PATH)
    parser.add_argument("--ltp-batch-size", type=int, default=DEFAULT_LTP_BATCH_SIZE)
    parser.add_argument("--ltp-device", choices=["auto", "cpu", "cuda"], default=DEFAULT_LTP_DEVICE)
    parser.add_argument("--ltp-physical-gpu", default=DEFAULT_LTP_PHYSICAL_GPU)
    args = parser.parse_args()

    build_stage2_outputs(
        input_path=args.input,
        output_dir=args.output_dir,
        manifest_name=args.manifest_name,
        ltp_model_path=args.ltp_model_path,
        ltp_batch_size=args.ltp_batch_size,
        ltp_device=args.ltp_device,
        ltp_physical_gpu=args.ltp_physical_gpu,
    )


if __name__ == "__main__":
    main()
