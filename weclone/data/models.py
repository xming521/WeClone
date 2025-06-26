from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pandas import Timestamp
from pydantic import BaseModel, Field

from weclone.utils.config_models import DataModality
from weclone.utils.i18n import MultiLangList


@dataclass
class ChatMessage:
    id: int  # 顺序id
    MsgSvrID: str  # 消息平台原始id
    type_name: str  # 消息类型 参考cut_type_data和skip_type_data
    is_sender: int  # 0: 对方 1: 自己
    talker: str  # 消息发送者
    msg: str  # 消息内容
    src: str  # 媒体文件路径、额外信息字段
    CreateTime: Timestamp  # 消息发送时间
    room_name: Optional[str] = None  # 聊天室名称
    is_forward: bool = False  # 是否是转发消息
    modality: Optional[DataModality] = None  # 消息模态  set in qa_generator.py


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
    score: int = Field(ge=1, le=5)


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
        "cut",
        "Cut",
        "image",
        "video",
        "merged forward chat records",
        "voice",
        "(share) music",
        "(share) card link",
        "(share) note",
        "(share) mini program",
        "(share) favorites",
        "(share) novel (guess)",
        "(share) video account card",
        "(share) video account video",
        "pasted text",  # Unparseable share link
    ],
}

cut_type_list = MultiLangList(cut_type_data, default_lang="en")


skip_type_data = {
    "zh_CN": [
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
    ],
    "en": [
        "add friend",
        "recommend official account",
        "animated emoji",
        "location",
        "file",
        "location sharing",
        "reply with quote",
        "group announcement",
        "transfer",
        "voice call",
        "system notification",
        "message recall",
        "invite to group",
    ],
}

skip_type_list = MultiLangList(skip_type_data, default_lang="en")

# 没处理的类型
unprocessed_type_list = []
