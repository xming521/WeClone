import json

import pandas as pd

from weclone.data.agent import chat_snapshot
from weclone.utils.config_models import MakeDatasetArgs

START = pd.Timestamp("2025-01-01 10:00:00")


def write_chat(root, folder, rows, remark=None):
    """rows: (minutes after START, is_sender, type_name, msg, forwarded)."""
    path = root / folder
    path.mkdir(parents=True)
    records = [
        {
            "id": index,
            "MsgSvrID": f"{folder}-{index}",
            "type_name": type_name,
            "is_sender": sender,
            "talker": "me" if sender else folder,
            "room_name": "",
            "msg": msg,
            "src": "",
            "CreateTime": (START + pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S"),
            "is_forward": forwarded,
        }
        for index, (minutes, sender, type_name, msg, forwarded) in enumerate(rows)
    ]
    pd.DataFrame(records).to_csv(path / f"{folder}_1_{len(rows)}.csv", index=False)
    if remark is not None:
        (path / "users.json").write_text(
            json.dumps({folder: {"remark": remark}}, ensure_ascii=False), encoding="utf-8"
        )


def snapshot(tmp_path, csv_root, name):
    out = tmp_path / name
    args = MakeDatasetArgs(platform="chat", csv_folder=str(csv_root))
    manifest = chat_snapshot.build_snapshot(out, args, registry_path=tmp_path / "contacts.json")
    return out, {contact["chat_with_id"]: contact for contact in manifest["contacts"]}


def text(minutes, sender, msg, forwarded=False):
    return (minutes, sender, "文本", msg, forwarded)


def test_snapshot_keeps_every_text_message_and_cuts_sessions_at_silences(tmp_path):
    write_chat(
        tmp_path / "csv",
        "wxid_a",
        [
            text(0, 0, "周末一起去爬山吗"),
            text(5, 0, "我在北京上学，周六有空"),  # unanswered before the next one, place name included
            text(6, 1, "好啊"),
            text(7.5, 1, "几点出发"),  # within two minutes of B's last message
            (7.8, 1, "图片", "图片", False),
            text(8, 1, "转发的一篇文章正文很长很长", forwarded=True),
            text(30, 1, "我到了"),  # after a silence, too short to keep
            text(60, 0, "下周三的会议改到下午三点了"),  # a session B never answered
        ],
        remark="老同学",
    )
    out, contacts = snapshot(tmp_path, tmp_path / "csv", "people")

    sessions = json.loads((out / "P1.json").read_text(encoding="utf-8"))
    assert [[(m["role"], m["content"]) for m in s["messages"]] for s in sessions] == [
        [("user", "周末一起去爬山吗"), ("user", "我在北京上学，周六有空"), ("assistant", "好啊\n几点出发")],
        [("user", "下周三的会议改到下午三点了")],
    ]
    first = sessions[0]
    assert (first["id"], first["chat_with"], first["chat_with_id"]) == ("0", "老同学", "P1")
    assert first["time"] == first["messages"][-1]["time"] == "2025-01-01T10:07:30"
    assert (contacts["P1"]["sessions"], contacts["P1"]["skipped_short_sessions"]) == (2, 1)


def test_snapshot_ids_come_from_the_registry_and_survive_new_exports(tmp_path):
    first = tmp_path / "csv-first"
    write_chat(first, "wxid_a", [text(60, 0, "明天下午一起打球吧"), text(61, 1, "好的没问题，几点")])
    write_chat(first, "wxid_b", [text(0, 0, "论文的初稿写完了吗"), text(1, 1, "还差最后一节实验")])
    out, contacts = snapshot(tmp_path, first, "people-first")
    # Two contacts without a remark stay apart, numbered by their first message.
    assert {cid: c["chat_with"] for cid, c in contacts.items()} == {"P1": "", "P2": ""}
    assert sorted(p.name for p in out.iterdir()) == ["P1.json", "P2.json", chat_snapshot.MANIFEST_NAME]
    assert (
        json.loads((out / "P1.json").read_text(encoding="utf-8"))[0]["messages"][0]["content"]
        == "论文的初稿写完了吗"
    )

    # A later export where a new contact comes first and a remark changed keeps the old ids.
    second = tmp_path / "csv-second"
    write_chat(second, "wxid_c", [text(-60, 0, "你好，我是新来的同事小李")], remark="新朋友")
    write_chat(
        second, "wxid_b", [text(0, 0, "论文的初稿写完了吗"), text(1, 1, "还差最后一节实验")], remark="老同学"
    )
    _, contacts = snapshot(tmp_path, second, "people-second")
    assert {cid: c["chat_with"] for cid, c in contacts.items()} == {"P1": "老同学", "P3": "新朋友"}
