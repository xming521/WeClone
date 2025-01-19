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