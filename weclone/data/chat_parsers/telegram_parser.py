import csv
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional

from pandas import Timestamp

from weclone.data.models import ChatMessage
from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger


class TelegramChatParser:
    """Telegram聊天记录解析器，将JSON格式转换为符合ChatMessage结构的数据"""

    def __init__(self, my_user_id: Optional[str] = None):
        """
        Constructor.

        Parameters
        ----------
        current_user_id : Optional[str]
            当前用户的from_id，用于判断是否为发送者。如果不提供，会自动分析
        """
        self.my_user_id = my_user_id
        self.message_counter = 0

        self.type_mapping = {
            "text": "text",
            "photo": "image",
            "video_file": "video",
            "animation": "video",  # GIF动画也归类为视频
            "voice_message": "voice",
            "audio_file": "file",
            "sticker": "animated emoji",
            "file": "file",
            "location": "location",
            "poll": "(share) card link",  # 投票
            "contact_information": "(share) card link",
        }

    def get_message_type_and_content(self, message: Dict) -> tuple[str, str, str, bool]:
        """
        根据Telegram消息内容确定type_name、msg内容、src和是否为转发消息

        Returns
        -------
        tuple[str, str, str, bool]
            (type_name, msg_content, src_path, is_forward)
        """
        msg_content = ""
        src_path = ""
        msg_type = "text"
        is_forward = "forwarded_from" in message

        if "text" in message:
            msg_content = self.extract_text_content(message["text"])

        if "media_type" in message:
            media_type = message["media_type"]
            msg_type = media_type

            if media_type == "photo":
                src_path = message.get("photo", "")
            elif media_type in ["video_file", "animation"]:
                src_path = message.get("file", "")
            elif media_type == "voice_message":
                src_path = message.get("file", "")
            elif media_type == "audio_file":
                src_path = message.get("file", "")
            elif media_type == "sticker":
                src_path = message.get("file", "")
                if not msg_content.strip():
                    msg_content = message.get("sticker_emoji", "")
            else:
                src_path = message.get("file", "")

        elif "photo" in message:
            msg_type = "photo"
            src_path = message["photo"]

        elif "file" in message:
            msg_type = "file"
            src_path = message["file"]
            if not msg_content.strip():
                msg_content = message.get("file_name", "")

        elif "location_information" in message:
            msg_type = "location"
            loc = message["location_information"]
            src_path = f"lat:{loc.get('latitude', 0)},lng:{loc.get('longitude', 0)}"
            if not msg_content.strip():
                msg_content = message.get("place_name", "") + message.get("address", "")

        type_name = self.type_mapping[msg_type]

        return type_name, msg_content.strip(), src_path, is_forward

    def extract_text_content(self, text_field) -> str:
        """从text字段提取纯文字内容"""
        content = ""
        if isinstance(text_field, str):
            content = text_field
        elif isinstance(text_field, list):
            for item in text_field:
                if isinstance(item, str):
                    content += item
                elif isinstance(item, dict) and "text" in item:
                    content += item["text"]

        return content.replace('\\"', "")

    def determine_sender_type(self, from_id: str) -> int:
        """确定发送者类型：0表示对方，1表示自己"""
        return 1 if from_id == self.my_user_id else 0

    def process_message(self, message: Dict) -> List[ChatMessage]:
        """
        处理单个消息，可能返回多条消息（原始消息+提取的文本消息）

        Parameters
        ----------
        message : Dict
            Telegram消息对象

        Returns
        -------
        List[ChatMessage]
            解析后的ChatMessage对象列表
        """
        if message.get("type") != "message":
            return []

        msg_id = message.get("id", 0)
        sender_name = message.get("from", "")
        from_id = message.get("from_id", "")
        date = message.get("date", "")

        type_name, msg_content, src_path, is_forward = self.get_message_type_and_content(message)

        try:
            dt = datetime.fromisoformat(date.replace("T", " ").replace("Z", ""))
            create_time = Timestamp(dt)
        except Exception as e:
            logger.warning(f"时间格式转换失败: {date}, 错误: {e}")
            create_time = Timestamp.now()

        is_sender = self.determine_sender_type(from_id)
        self.message_counter += 1

        result_messages = []
        # 对于有内容的消息或有媒体文件的消息，都要保存
        if msg_content.strip() or src_path.strip():
            original_msg = ChatMessage(
                id=self.message_counter,  # 使用全局计数作为顺序ID
                MsgSvrID=msg_id,  # Telegram消息ID
                type_name=type_name,
                is_sender=is_sender,  # 0: 对方 1: 自己
                talker=sender_name,
                msg=msg_content.replace("\n", " ").strip() if msg_content.strip() else f"{type_name}",
                src=src_path,
                CreateTime=create_time,
                is_forward=is_forward,
            )
            result_messages.append(original_msg)

        # 如果是非纯文本消息但包含text字段，创建额外的文本消息
        if type_name not in ["text"] and "text" in message:
            text_content = self.extract_text_content(message["text"])
            if text_content.strip():
                self.message_counter += 1
                text_msg = ChatMessage(
                    id=self.message_counter,
                    MsgSvrID=msg_id,
                    type_name="text",
                    is_sender=is_sender,
                    talker=sender_name,
                    msg=text_content.replace("\n", " ").strip(),
                    src=f"from_msg_id:{msg_id}",
                    CreateTime=create_time,
                    is_forward=is_forward,
                )
                result_messages.append(text_msg)

        return result_messages

    def process_chat(self, jdata: Dict) -> List[ChatMessage]:
        """
        处理聊天数据

        Parameters
        ----------
        jdata : Dict
            Telegram聊天JSON对象

        Returns
        -------
        List[ChatMessage]
            ChatMessage对象列表
        """
        chat_name = jdata.get("name", "未知聊天")
        messages = jdata.get("messages", [])

        chat_messages = []
        for message in messages:
            chat_msgs = self.process_message(message)
            chat_messages.extend(chat_msgs)

        for msg in chat_messages:
            msg.room_name = chat_name

        logger.info(f"聊天 '{chat_name}' 解析完成，共{len(chat_messages)}条消息")
        return chat_messages

    def to_csv(self, chat_messages: List[ChatMessage], output_file: str):
        """
        保存ChatMessage列表到CSV文件

        Parameters
        ----------
        chat_messages : List[ChatMessage]
            ChatMessage对象列表
        output_file : str
            输出CSV文件路径
        """
        if not chat_messages:
            logger.warning("没有消息需要保存")
            return

        # 定义CSV列名
        fieldnames = [
            "id",  # 顺序id
            "MsgSvrID",  # 消息服务器ID
            "type_name",  # 消息类型名称
            "is_sender",  # 0: 对方 1: 自己
            "talker",  # 发言人
            "room_name",  # 聊天室名称
            "msg",  # 消息内容
            "src",  # 消息来源/媒体文件路径
            "CreateTime",  # 创建时间
            "is_forward",  # 是否为转发消息
        ]

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for msg in chat_messages:
                writer.writerow(
                    {
                        "id": msg.id,
                        "MsgSvrID": msg.MsgSvrID,
                        "type_name": msg.type_name,
                        "is_sender": msg.is_sender,
                        "talker": msg.talker,
                        "room_name": msg.room_name,
                        "msg": msg.msg,
                        "src": msg.src,
                        "CreateTime": msg.CreateTime,
                        "is_forward": msg.is_forward,
                    }
                )

        logger.info(f"CSV文件已保存: {output_file}")

    def copy_received_images(
        self, chat_messages: List[ChatMessage], base_path: str = "", target_dir: str = "dataset/media/images"
    ):
        """
        复制所有is_sender为0的图片到指定目录

        Parameters
        ----------
        chat_messages : List[ChatMessage]
            ChatMessage对象列表
        base_path : str
            图片文件的基础路径前缀
        target_dir : str
            目标目录，默认为dataset/media/images
        """
        os.makedirs(target_dir, exist_ok=True)

        copied_count = 0
        skipped_count = 0

        for msg in chat_messages:
            if msg.is_sender == 0 and msg.type_name == "image" and msg.src:
                if base_path:
                    full_src_path = os.path.join(base_path, msg.src)
                else:
                    full_src_path = msg.src

                normalized_src = full_src_path.replace("\\", "/")
                if not os.path.exists(normalized_src):
                    logger.warning(f"源文件不存在: {normalized_src}")
                    skipped_count += 1
                    continue

                filename = os.path.basename(normalized_src)

                target_path = os.path.join(target_dir, filename)

                shutil.copy2(normalized_src, target_path)
                copied_count += 1

        logger.info(f"图片复制完成: 成功 {copied_count}, 跳过 {skipped_count}")


def process_telegram_dataset(config: WCMakeDatasetConfig) -> None:
    """
    处理Telegram数据集，遍历dataset/telegram下的所有文件夹
    为每个telegram文件夹在dataset/csv下创建对应的文件夹

    Parameters
    ----------
    config : WCMakeDatasetConfig
        数据集配置，包含telegram_args.my_id用于判断发送者
    """
    telegram_dir = "dataset/telegram"
    csv_output_dir = "dataset/csv"

    if not os.path.exists(telegram_dir):
        logger.error(f"Telegram数据目录不存在: {telegram_dir}")
        return

    if not config.telegram_args or not config.telegram_args.my_id:
        logger.error("Telegram配置缺失，无法处理Telegram数据集")
        sys.exit(1)

    if os.path.exists(csv_output_dir):
        for item in os.listdir(csv_output_dir):
            item_path = os.path.join(csv_output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    my_id = config.telegram_args.my_id

    for folder_name in os.listdir(telegram_dir):
        folder_path = os.path.join(telegram_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "result.json")
        if not os.path.exists(json_path):
            logger.warning(f"文件夹 {folder_name} 中未找到result.json文件")
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            jdata = json.load(file)

        chat_name = jdata.get("name", "unknown")
        chat_type = jdata.get("type", "unknown")
        chat_id = jdata.get("id", "unknown")

        safe_name = "".join(c for c in str(chat_name) if c.isalnum() or c in "._-")
        safe_type = "".join(c for c in str(chat_type) if c.isalnum() or c in "._-")
        safe_id = "".join(c for c in str(chat_id) if c.isalnum() or c in "._-")

        csv_folder_name = f"{safe_name}-{safe_type}-{safe_id}"
        csv_folder_path = os.path.join(csv_output_dir, csv_folder_name)

        parser = TelegramChatParser(my_user_id=my_id)
        messages = parser.process_chat(jdata)

        if messages:
            csv_file_path = os.path.join(csv_folder_path, f"{csv_folder_name}.csv")
            parser.to_csv(messages, csv_file_path)
            parser.copy_received_images(messages, folder_path)
        else:
            logger.warning(f"文件夹 '{folder_name}' 没有有效消息")
