from dataclasses import dataclass
from enum import Enum

from pandas import Timestamp
from pydantic import BaseModel

from weclone.utils.i18n import MultiLangList


@dataclass
class ChatMessage:
    id: int
    MsgSvrID: int
    type_name: str
    is_sender: int  # 0: 对方 1: 自己
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


class QaPairFormat(Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class QaPair:
    """支持sharegpt格式的QA对类"""

    id: int
    time: Timestamp
    score: int
    messages: list[Message]
    images: list[str]
    system: str
    format_type: QaPairFormat = QaPairFormat.SHAREGPT
    # data: Union[AlpacaQaPair, ShareGPTQaPair]


class QaPairScore(BaseModel):
    id: int
    score: int


cut_type_data = {
    "zh_CN": [
        "Cut",
        "图片",
        "视频",
        "合并转发的聊天记录",
        "语音",
        "(分享)音乐",
        "(分享)卡片式链接",
        "(分享)笔记",
        "(分享)小程序",
        "(分享)收藏夹",
        "(分享)小说(猜)",
        "(分享)视频号名片",
        "(分享)视频号视频",
        "粘贴的文本",  # 无法解析的分享链接
        "未知",
    ],
    "en": [
        "Cut",
        "Image",
        "Video",
        "Merged Forward Chat Records",
        "Voice",
        "(Share) Music",
        "(Share) Card Link",
        "(Share) Note",
        "(Share) Mini Program",
        "(Share) Favorites",
        "(Share) Novel (Guess)",
        "(Share) Video Account Card",
        "(Share) Video Account Video",
        "Pasted Text",  # Unparseable share link
    ],
}

cut_type_list = MultiLangList(cut_type_data, default_lang="en")

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
