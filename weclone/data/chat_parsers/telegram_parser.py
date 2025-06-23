import csv
import json
import os
import sys
from collections import Counter
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
        self.message_counter = 0  # 全局消息计数器

        # Telegram媒体类型到cut_type_data en列表的映射
        self.type_mapping = {
            "text": "Text",
            "photo": "Image",
            "video_file": "Video",
            "animation": "Video",  # GIF动画也归类为视频
            "voice_message": "Voice",
            "audio_file": "File",
            "sticker": "Animated Emoji",
            "file": "File",  # 普通文件
            "location": "Location",  # 位置信息
            "forwarded": "Merged Forward Chat Records",  # 转发消息
            "poll": "(Share) Card Link",  # 投票
            "contact_information": "(Share) Card Link",  # 联系人
        }

    def get_message_type_and_content(self, message: Dict) -> tuple[str, str, str]:
        """
        根据Telegram消息内容确定type_name、msg内容和src

        Returns
        -------
        tuple[str, str, str]
            (type_name, msg_content, src_path)
        """
        msg_content = ""
        src_path = ""
        msg_type = "text"  # 默认为文本

        # 首先尝试从text字段获取内容
        if "text" in message:
            msg_content = self.extract_text_content(message["text"])

        # 检查是否为转发消息
        if "forwarded_from" in message:
            msg_type = "forwarded"
            # 保持原有的text内容，不添加额外描述

        # 检查媒体类型
        elif "media_type" in message:
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
                # 如果贴纸没有text内容，使用emoji作为内容
                if not msg_content.strip():
                    msg_content = message.get("sticker_emoji", "")
            else:
                src_path = message.get("file", "")

        # 检查是否有单独的photo字段（非media_type的图片）
        elif "photo" in message:
            msg_type = "photo"
            src_path = message["photo"]
            # text内容已经在上面提取了

        # 检查是否有文件
        elif "file" in message:
            msg_type = "file"
            src_path = message["file"]
            # 如果没有text内容，使用文件名
            if not msg_content.strip():
                msg_content = message.get("file_name", "")

        # 检查是否有位置信息
        elif "location_information" in message:
            msg_type = "location"
            loc = message["location_information"]
            src_path = f"lat:{loc.get('latitude', 0)},lng:{loc.get('longitude', 0)}"
            # 如果没有text内容，使用地址信息
            if not msg_content.strip():
                address = message.get("address", "")
                place_name = message.get("place_name", "")
                msg_content = place_name if place_name else address if address else ""

        # 映射到cut_type_data的en类型
        type_name = self.type_mapping.get(msg_type, "Pasted Text")

        return type_name, msg_content.strip(), src_path

    def extract_text_content(self, text_field) -> str:
        """从text字段提取纯文字内容"""
        if isinstance(text_field, str):
            return text_field
        elif isinstance(text_field, list):
            content = ""
            for item in text_field:
                if isinstance(item, str):
                    content += item
                elif isinstance(item, dict) and "text" in item:
                    content += item["text"]
            return content
        return ""

    def auto_detect_current_user(self, messages: List[Dict]) -> Optional[str]:
        """
        自动检测当前用户ID
        通常导出聊天记录的用户就是当前用户，可以通过分析消息分布来判断
        """
        if not messages:
            return None

        # 统计所有from_id的出现频率
        from_id_counter = Counter()
        for msg in messages:
            if msg.get("type") == "message" and "from_id" in msg:
                from_id_counter[msg["from_id"]] += 1

        if not from_id_counter:
            return None

        # 返回出现频率最高的from_id作为当前用户
        most_common_user = from_id_counter.most_common(1)[0][0]
        logger.info(f"自动检测到当前用户ID: {most_common_user}")
        return most_common_user

    def determine_sender_type(self, from_id: str) -> int:
        """确定发送者类型：0表示对方，1表示自己"""
        if self.my_user_id is None:
            return 0  # 如果无法确定当前用户，默认为对方
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
        # 只处理message类型的消息
        if message.get("type") != "message":
            return []

        # 提取基本信息
        msg_id = message.get("id", 0)
        sender_name = message.get("from", "")
        from_id = message.get("from_id", "")
        date = message.get("date", "")

        # 获取消息类型、内容和源文件路径
        type_name, msg_content, src_path = self.get_message_type_and_content(message)

        # 转换时间格式
        try:
            # Telegram的date格式是ISO 8601
            dt = datetime.fromisoformat(date.replace("T", " ").replace("Z", ""))
            create_time = Timestamp(dt)
        except Exception as e:
            logger.warning(f"时间格式转换失败: {date}, 错误: {e}")
            create_time = Timestamp.now()

        # 确定发送者类型
        is_sender = self.determine_sender_type(from_id)

        # 增加全局消息计数
        self.message_counter += 1

        result_messages = []

        # 创建原始消息（如果有有效内容或者是媒体消息）
        if msg_content.strip() or type_name != "Pasted Text":
            original_msg = ChatMessage(
                id=self.message_counter,  # 使用全局计数作为顺序ID
                MsgSvrID=msg_id,  # Telegram消息ID
                type_name=type_name,  # 映射到cut_type_data的类型
                is_sender=is_sender,  # 0: 对方 1: 自己
                talker=sender_name,  # 发言人
                msg=msg_content.replace("\n", " ").strip(),  # 消息内容
                src=src_path,  # 媒体文件路径或其他源信息
                CreateTime=create_time,  # 创建时间
            )
            result_messages.append(original_msg)

        # 如果是非纯文本消息但包含text字段，创建额外的文本消息
        if type_name != "Text" and "text" in message:
            text_content = self.extract_text_content(message["text"])
            if text_content.strip():
                self.message_counter += 1
                text_msg = ChatMessage(
                    id=self.message_counter,  # 使用全局计数作为顺序ID
                    MsgSvrID=msg_id,  # 使用相同的Telegram消息ID
                    type_name="Text",  # 文本消息类型
                    is_sender=is_sender,  # 0: 对方 1: 自己
                    talker=sender_name,  # 发言人
                    msg=text_content.replace("\n", " ").strip(),  # 文本内容
                    src=f"from_msg_id:{msg_id}",  # 来源消息ID
                    CreateTime=create_time,  # 创建时间
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

        # 如果还没有设置当前用户ID，尝试自动检测
        if self.my_user_id is None:
            self.my_user_id = self.auto_detect_current_user(messages)

        chat_messages = []
        for message in messages:
            chat_msgs = self.process_message(message)
            chat_messages.extend(chat_msgs)

        # 统一设置room_name
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
                    }
                )

        logger.info(f"CSV文件已保存: {output_file}")


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

    my_id = config.telegram_args.my_id

    for folder_name in os.listdir(telegram_dir):
        folder_path = os.path.join(telegram_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 查找result.json文件
        json_path = os.path.join(folder_path, "result.json")
        if not os.path.exists(json_path):
            logger.warning(f"文件夹 {folder_name} 中未找到result.json文件")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                jdata = json.load(file)

            # 提取聊天信息用于命名
            chat_name = jdata.get("name", "unknown")
            chat_type = jdata.get("type", "unknown")
            chat_id = jdata.get("id", "unknown")

            # 清理名称中的特殊字符
            safe_name = "".join(c for c in str(chat_name) if c.isalnum() or c in "._-")
            safe_type = "".join(c for c in str(chat_type) if c.isalnum() or c in "._-")
            safe_id = "".join(c for c in str(chat_id) if c.isalnum() or c in "._-")

            # 创建文件夹名：name-type-id
            csv_folder_name = f"{safe_name}-{safe_type}-{safe_id}"
            csv_folder_path = os.path.join(csv_output_dir, csv_folder_name)

            # 处理聊天消息
            parser = TelegramChatParser(my_user_id=my_id)
            messages = parser.process_chat(jdata)

            # 保存CSV文件
            if messages:
                csv_file_path = os.path.join(csv_folder_path, f"{csv_folder_name}.csv")
                parser.to_csv(messages, csv_file_path)
                logger.info(f"文件夹 '{folder_name}' 处理完成，共{len(messages)}条消息")
            else:
                logger.warning(f"文件夹 '{folder_name}' 没有有效消息")

        except Exception as e:
            logger.error(f"处理文件夹失败 {folder_path}: {e}")
