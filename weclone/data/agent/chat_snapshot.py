"""build-profile's chat snapshot: the text messages of every chat, grouped by contact and cut into sessions.

Profile and event extraction and style read this instead of the SFT dataset. Nothing is dropped for the training
format or for privacy: facts are kept and get a privacy level later, and disclosure happens at export and runtime.
Non-text messages are skipped.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyjson5

from weclone.data import chat_csv
from weclone.data.agent.contacts import DEFAULT_REGISTRY_PATH, assign_contact_ids, contact_key
from weclone.data.chat_parsers.telegram_parser import process_telegram_dataset
from weclone.utils import secure_storage
from weclone.utils.config_models import MakeDatasetArgs, PlatformType
from weclone.utils.log import logger

TEXT_TYPES = {"文本", "text"}
MERGE_SECONDS = 120  # one sender's messages this close together become one message, joined by newlines
SESSION_GAP_SECONDS = 600  # a longer silence starts a new session
MIN_SESSION_CHARS = 10  # shorter sessions (non-space characters) are left out to save extraction calls
MANIFEST_NAME = "manifest.json"


def load_chat_args(config_path: Path) -> MakeDatasetArgs:
    """Where the chats are: platform, csv_folder and telegram_args of make_dataset_args."""
    config = pyjson5.loads(Path(config_path).read_text(encoding="utf-8"))
    return MakeDatasetArgs(**config["make_dataset_args"])


def read_chats(chat_args: MakeDatasetArgs) -> dict[str, dict]:
    """The remark and text messages of each contact, keyed by contact key."""
    parsed = process_telegram_dataset(chat_args) if chat_args.platform == PlatformType.TELEGRAM else None
    folders: dict[Path, list[str]] = defaultdict(list)
    for path in chat_csv.list_csv_files(chat_args.csv_folder, parsed):
        folders[Path(path).parent].append(path)
    chats: dict[str, dict] = {}
    for folder, files in folders.items():
        rows = pd.concat([chat_csv.read_csv(path) for path in files], ignore_index=True)
        # Resolved over the whole folder: a chunk with only B's messages would fall back to the folder name.
        talker = chat_csv.find_partner_talker(folder.name, rows)
        partner = chat_csv.load_users_data(folder).get(talker)
        remark = str(partner.get("remark", "")).strip() if isinstance(partner, dict) else ""
        rows = rows[
            rows["type_name"].astype(str).str.lower().isin(TEXT_TYPES) & (rows["msg"].str.strip() != "")
        ]
        if "is_forward" in rows.columns:
            rows = rows[~((rows["is_sender"] == 1) & (rows["is_forward"]))]
        chat = chats.setdefault(contact_key(str(chat_args.platform), talker), {"chat_with": "", "rows": []})
        chat["chat_with"] = chat["chat_with"] or remark
        chat["rows"].append(rows)
    return chats


def sessions(rows: pd.DataFrame) -> list[list[dict]]:
    """Messages in time order, cut where the chat went quiet, with each sender's bursts merged."""
    rows = rows.assign(CreateTime=pd.to_datetime(rows["CreateTime"])).sort_values("CreateTime", kind="stable")
    result: list[list[dict]] = []
    previous = None
    for row in rows.itertuples(index=False):
        gap = (row.CreateTime - previous.CreateTime).total_seconds() if previous is not None else None
        if gap is None or gap > SESSION_GAP_SECONDS:
            result.append([])
        messages = result[-1]
        role = "assistant" if int(row.is_sender) == 1 else "user"
        # A merged message carries the time of its last part.
        if (
            messages
            and messages[-1]["role"] == role
            and row.talker == previous.talker
            and gap <= MERGE_SECONDS
        ):
            messages[-1]["content"] += "\n" + row.msg
            messages[-1]["time"] = row.CreateTime.isoformat()
        else:
            messages.append({"role": role, "content": row.msg, "time": row.CreateTime.isoformat()})
        previous = row
    return result


def build_snapshot(
    output_dir: Path, chat_args: MakeDatasetArgs, *, registry_path: Path | str = DEFAULT_REGISTRY_PATH
) -> dict:
    """Write one P{n}.json per contact (its sessions in time order) and a manifest into output_dir."""
    secure_storage.ensure_unlocked()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Snapshot directory is not empty: {output_dir}")
    contacts: dict[str, tuple[str, list[list[dict]], int]] = {}
    for key, chat in read_chats(chat_args).items():
        # The same chat exported into two folders must not count twice.
        rows = pd.concat(chat["rows"], ignore_index=True).drop_duplicates(
            subset=["MsgSvrID", "CreateTime", "is_sender", "msg"]
        )
        found = sessions(rows)
        kept = [s for s in found if sum(len("".join(m["content"].split())) for m in s) >= MIN_SESSION_CHARS]
        if kept:
            contacts[key] = (chat["chat_with"], kept, len(found) - len(kept))
    # New contacts are numbered in the order they first appear.
    ids = assign_contact_ids(
        sorted(contacts, key=lambda key: (contacts[key][1][0][0]["time"], key)), registry_path
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for key in sorted(contacts, key=lambda key: int(ids[key][1:])):
        chat_with, kept, skipped = contacts[key]
        items = [
            {
                "id": str(index),
                "time": session[-1]["time"],
                "chat_with": chat_with,
                "chat_with_id": ids[key],
                "messages": session,
            }
            for index, session in enumerate(kept)
        ]
        secure_storage.write_json(output_dir / f"{ids[key]}.json", items, indent=4)
        summary.append(
            {
                "chat_with_id": ids[key],
                "chat_with": chat_with,
                "sessions": len(items),
                "messages": sum(len(session) for session in kept),
                "skipped_short_sessions": skipped,
            }
        )
    manifest = {
        "csv_folder": str(chat_args.csv_folder),
        "platform": str(chat_args.platform),
        "merge_seconds": MERGE_SECONDS,
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "min_session_chars": MIN_SESSION_CHARS,
        "contacts": summary,
    }
    secure_storage.write_json(output_dir / MANIFEST_NAME, manifest)
    logger.info(
        f"Chat snapshot: {len(summary)} contacts, {sum(c['sessions'] for c in summary)} sessions "
        f"({sum(c['skipped_short_sessions'] for c in summary)} short ones skipped), saved to {output_dir}"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the chat snapshot that profile extraction reads.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("settings.jsonc"))
    parser.add_argument("--contacts", type=Path, default=DEFAULT_REGISTRY_PATH, help="Contact id registry")
    args = parser.parse_args(argv)
    secure_storage.configure(args.config_path)
    build_snapshot(args.output_dir, load_chat_args(args.config_path), registry_path=args.contacts)


if __name__ == "__main__":
    main()
