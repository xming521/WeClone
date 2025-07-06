import csv
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List

from pandas import Timestamp

from weclone.data.models import ChatMessage
from weclone.utils.config_models import DataModality, WCMakeDatasetConfig
from weclone.utils.log import logger


class TelegramChatParser:
    """Telegram chat parser that converts JSON format to data conforming to ChatMessage structure"""

    def __init__(self, config: WCMakeDatasetConfig):
        self.config = config
        self.my_user_id = config.telegram_args.my_id if config.telegram_args else None
        self.message_counter = 0

        self.type_mapping = {
            "text": "text",
            "photo": "image",
            "video_file": "video",
            "animation": "video",
            "voice_message": "voice",
            "audio_file": "file",
            "sticker": "sticker",
            "file": "file",
            "location": "location",
            "poll": "(share) card link",
            "contact_information": "(share) card link",
        }

    def get_message_type_and_content(self, message: Dict) -> tuple[str, str, str, bool]:
        """
        Determine type_name, msg content, src and whether it's a forwarded message based on Telegram message content

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
                # Only set sticker emoji as msg_content if STICKER is in include_type
                if DataModality.STICKER in self.config.include_type and not msg_content.strip():
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
        return 1 if from_id == self.my_user_id else 0

    def process_message(self, message: Dict) -> List[ChatMessage]:
        """
        Process a single message, may return multiple messages (original message + extracted text message)
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
            logger.warning(f"Time format conversion failed: {date}, error: {e}")

        is_sender = self.determine_sender_type(from_id)
        self.message_counter += 1

        result_messages = []
        # Save messages with content or media files
        if msg_content.strip() or src_path.strip():
            original_msg = ChatMessage(
                id=self.message_counter,  # Use global counter as sequential ID
                MsgSvrID=msg_id,  # Telegram message ID
                type_name=type_name,
                is_sender=is_sender,  # 0: other party 1: myself
                talker=sender_name,
                msg=msg_content.replace("\n", " ").strip() if msg_content.strip() else f"{type_name}",
                src=src_path,
                CreateTime=create_time,
                is_forward=is_forward,
            )
            result_messages.append(original_msg)

        # If it's a non-pure text message but contains text field, create additional text message
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
                    src="",
                    CreateTime=create_time,
                    is_forward=is_forward,
                )
                result_messages.append(text_msg)

        return result_messages

    def process_chat(self, jdata: Dict) -> List[ChatMessage]:
        """
        Process chat data

        Parameters
        ----------
        jdata : Dict
            Telegram chat JSON object

        Returns
        -------
        List[ChatMessage]
            List of ChatMessage objects
        """
        chat_name = jdata.get("name", "Unknown Chat")
        messages = jdata.get("messages", [])

        chat_messages = []
        for message in messages:
            chat_msgs = self.process_message(message)
            chat_messages.extend(chat_msgs)

        for msg in chat_messages:
            msg.room_name = chat_name

        logger.info(f"Chat '{chat_name}' parsing completed, {len(chat_messages)} messages in total")
        return chat_messages

    def to_csv(self, chat_messages: List[ChatMessage], output_file: str):
        """
        Save ChatMessage list to CSV file

        Parameters
        ----------
        chat_messages : List[ChatMessage]
            List of ChatMessage objects
        output_file : str
            Output CSV file path
        """
        if not chat_messages:
            logger.warning("No messages to save")
            return

        fieldnames = [
            "id",
            "MsgSvrID",
            "type_name",
            "is_sender",
            "talker",
            "room_name",
            "msg",
            "src",
            "CreateTime",
            "is_forward",
        ]

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

        logger.info(f"CSV file saved: {output_file}")

    def copy_received_images(
        self, chat_messages: List[ChatMessage], base_path: str = "", target_dir: str = "dataset/media/images"
    ):
        """
        Copy all images with is_sender=0 to specified directory
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
                    logger.warning(f"Source file does not exist: {normalized_src}")
                    skipped_count += 1
                    continue

                filename = os.path.basename(normalized_src)

                target_path = os.path.join(target_dir, filename)

                shutil.copy2(normalized_src, target_path)
                copied_count += 1

        logger.info(f"Image copying completed: successful {copied_count}, skipped {skipped_count}")


def process_telegram_dataset(config: WCMakeDatasetConfig) -> None:
    """
    Process Telegram dataset, traverse all folders under dataset/telegram
    Create corresponding folders for each telegram folder under dataset/csv

    Parameters
    ----------
    config : WCMakeDatasetConfig
        Dataset configuration, contains telegram_args.my_id for determining sender
    """
    telegram_dir = "dataset/telegram"
    csv_output_dir = "dataset/csv"

    if not os.path.exists(telegram_dir):
        logger.error(f"Telegram data directory does not exist: {telegram_dir}")
        return

    if not config.telegram_args or not config.telegram_args.my_id:
        logger.error("Telegram configuration missing, cannot process Telegram dataset")
        sys.exit(1)

    if os.path.exists(csv_output_dir):
        for item in os.listdir(csv_output_dir):
            item_path = os.path.join(csv_output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    for folder_name in os.listdir(telegram_dir):
        folder_path = os.path.join(telegram_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "result.json")

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

        parser = TelegramChatParser(config=config)
        messages = parser.process_chat(jdata)

        if messages:
            csv_file_path = os.path.join(csv_folder_path, f"{csv_folder_name}.csv")
            parser.to_csv(messages, csv_file_path)
            parser.copy_received_images(messages, folder_path)
        else:
            logger.warning(f"Folder '{folder_name}' has no valid messages")
