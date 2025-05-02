from dataclasses import dataclass
from pandas import Timestamp


@dataclass
class ChatMessage:
    id: int
    MsgSvrID: int
    type_name: str
    is_sender: int
    talker: str
    room_name: str
    msg: str
    src: str
    CreateTime: Timestamp


@dataclass
class CutMessage:
    is_sender: int
    cut_type: str
    CreateTime: Timestamp


@dataclass
class QaPair:
    id: int
    system: str
    instruction: str
    output: str
    history: list[list[str]]
    time: Timestamp
    score: int


skip_type_list = [
    "添加好友",
    "推荐公众号",
    "动画表情",
    "位置",
    "文件",
    "位置共享",
    "接龙",
    "引用回复",
    "视频号直播或直播回放",
    "用户上传的GIF表情",
    "文件(猜)",
    "群公告",
    "视频号直播或直播回放等",
    "游戏相关",
    "转账",
    "赠送红包封面",
    "语音通话",
    "企业微信打招呼(猜)",
    "企业微信添加好友(猜)",
    "系统通知",
    "消息撤回1",
    "拍一拍",
    "消息撤回5",
    "消息撤回6",
    "消息撤回33",
    "消息撤回36",
    "消息撤回57",
    "邀请加群",
    "未知-11000,0",
]
# 没处理的类型
unprocessed_type_list = []
