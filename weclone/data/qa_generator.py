import json
import os
import re
import subprocess  # nosec
import sys
from typing import List, Union, cast

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
from pandas import Timestamp

from weclone.core.PII.pii_detector import ChinesePIIDetector, PIIDetector
from weclone.data.chat_parsers.telegram_parser import process_telegram_dataset
from weclone.data.clean.strategies import LLMCleaningStrategy, OlineLLMCleaningStrategy
from weclone.data.models import (
    ChatMessage,
    CutMessage,
    Message,
    QaPair,
    cut_type_list,
    skip_type_list,
)
from weclone.data.strategies import TimeWindowStrategy
from weclone.data.utils import ImageToTextProcessor, check_image_file_exists
from weclone.utils.config import load_config
from weclone.utils.config_models import DataModality, LanguageType, PlatformType, WCMakeDatasetConfig
from weclone.utils.log import logger


class DataProcessor:
    def __init__(self):
        self.config = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))
        self.csv_folder = "./dataset/csv"
        self.system_prompt = self.config.default_system
        self.enable_clean = self.config.clean_dataset.enable_clean

        # message type
        self.QaPair = QaPair

        self.include_type = self.config.include_type
        if self.config.platform == PlatformType.WECHAT:
            self.cut_type_list = cut_type_list.get_items(lang="zh_CN")
            self.skip_type_list = skip_type_list.get_items(lang="zh_CN")
            self.include_type = cut_type_list.translate_batch(
                texts=[t for t in self.include_type if t.lower() != "text"]
            )
            self.cut_type_list = [t for t in self.cut_type_list if t not in self.include_type]
        elif self.config.platform == PlatformType.TELEGRAM:
            self.cut_type_list = cut_type_list.get_items(lang="en")
            self.skip_type_list = skip_type_list.get_items(lang="en")
            self.include_type = [t for t in self.include_type if t.lower() != "text"]
            self.cut_type_list = [t for t in self.cut_type_list if t not in self.include_type]
            if DataModality.STICKER in self.include_type:
                self.skip_type_list.remove("sticker")

        # blocked words
        config_blocked_words = self.config.blocked_words
        file_blocked_words = []
        try:
            with open("./dataset/blocked_words.json", encoding="utf-8") as f:
                file_blocked_words = json.load(f).get("blocked_words", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        self.blocked_words = list(set(config_blocked_words + file_blocked_words))
        # logger.info(f"Chat record blocked words: {self.blocked_words}")

        # combine strategy
        if self.config.single_combine_strategy == "time_window":
            self.single_combine_strategy = TimeWindowStrategy(
                time_window=self.config.single_combine_time_window * 60,
                is_single_chat=True,
            )

        if self.config.qa_match_strategy == "time_window":
            self.qa_match_strategy = TimeWindowStrategy(
                time_window=self.config.qa_match_time_window * 60,
                is_single_chat=False,
            )

        # PII detection
        if self.config.language == LanguageType.ZH:
            self.pii_detector = ChinesePIIDetector()
        else:
            self.pii_detector = PIIDetector(language=self.config.language)

        # dataset cleaning
        clean_dataset_config = self.config.clean_dataset

        if self.enable_clean:
            if DataModality.IMAGE in self.config.include_type:
                logger.error("Enabling clean_dataset does not support image type messages")
                exit()

            if clean_dataset_config.clean_strategy == "llm":
                if self.config.online_llm_clear:
                    self.clean_strategy = OlineLLMCleaningStrategy(make_dataset_config=self.config)
                else:
                    from llamafactory.extras.packages import is_vllm_available

                    if not is_vllm_available():
                        logger.warning("vLLM is not available, dataset cleaning is temporarily disabled.")
                        self.enable_clean = False
                    else:
                        self.clean_strategy = LLMCleaningStrategy(make_dataset_config=self.config)

        vision_config = self.config.vision_api
        if vision_config.enable and vision_config.api_key:
            self.image_processor = ImageToTextProcessor(
                api_url=vision_config.api_url,  # type: ignore
                api_key=vision_config.api_key,  # type: ignore
                model_name=vision_config.model_name,  # type: ignore
                config=self.config,
            )
            logger.info(f"ImageToText functionality enabled, model: {self.image_processor.model_name}")
        else:
            self.image_processor = None

        self.c = self.config

    def main(self):
        self.pre_parse_chat_dataset()

        if not os.path.exists(self.csv_folder) or not os.listdir(self.csv_folder):
            logger.error(
                f"Error: Directory '{self.csv_folder}' does not exist or is empty. Please check the path and ensure it contains CSV chat data files."
            )
            sys.exit(1)

        csv_files = self.get_csv_files()
        logger.info(f"Found {len(csv_files)} CSV files in total, starting processing, please be patient...")
        message_list: List[ChatMessage] = []
        for csv_file in csv_files:
            logger.debug(f"Starting to process CSV file: {csv_file}")
            chat_messages = self.load_csv(csv_file)
            message_list.extend(self.group_consecutive_messages(messages=chat_messages))
            # self.process_by_msgtype(chat_message)
            logger.debug(f"Processing completed: {csv_file}, loaded {len(chat_messages)} messages in total")
        qa_res = self.match_qa(message_list)
        qa_res = [item for item in qa_res if isinstance(item, QaPair)]

        if self.image_processor:
            logger.info("Starting image recognition process...")
            qa_res = self.image_processor._process_images_in_parallel(qa_res)
            logger.info("Image recognition process completed.")

        if self.enable_clean:
            self.clean_strategy.judge(qa_res)  # type: ignore

        self.save_result(qa_res)
        self._execute_length_cdf_script()

        logger.success(
            f"Chat record processing successful, obtained {len(qa_res)} data entries in total, saved to ./dataset/res_csv/sft/sft-my.json"
        )

    def pre_parse_chat_dataset(self):
        if self.c.platform == PlatformType.TELEGRAM:
            process_telegram_dataset(self.config)

    def _execute_length_cdf_script(self):
        """Execute the length_cdf.py script to calculate cutoff_len."""
        try:
            python_executable = sys.executable
            script_path = os.path.join("weclone", "utils", "length_cdf.py")

            command_parts = [
                python_executable,
                script_path,
                f'--model_name_or_path="{self.c.model_name_or_path}"',
                f'--dataset="{self.c.dataset}"',
                f'--dataset_dir="{self.c.dataset_dir}"',
                f'--template="{self.c.template}"',
                "--interval=512",
            ]

            if hasattr(self.c, "media_dir") and self.c.media_dir:
                command_parts.append(f'--media_dir="{self.c.media_dir}"')
            if hasattr(self.c, "image_max_pixels") and self.c.image_max_pixels:
                command_parts.append(f'--image_max_pixels="{self.c.image_max_pixels}"')

            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = "0"
            child_env["LLAMAFACTORY_VERBOSITY"] = "ERROR"

            process = subprocess.Popen(
                command_parts,
                env=child_env,
                stdout=None,  # Use None to indicate using parent process's stdout (i.e., terminal)
                stderr=None,
                text=True,
                bufsize=1,
            )  # nosec
            return_code = process.wait()
            if return_code != 0:
                logger.error(
                    f"Command '{' '.join(command_parts)}' execution failed with return code {return_code}"
                )
        except FileNotFoundError:
            logger.error(
                f"Command execution failed: executable '{command_parts[0]}' or script '{command_parts[1]}' not found"
            )
        except KeyError as e:
            logger.error(f"Failed to execute length_cdf.py script: missing configuration item {str(e)}")
        except Exception as e:
            logger.error(f"Unknown error occurred while executing length_cdf.py script: {str(e)}")

    def get_csv_files(self):
        """Traverse the folder to get all CSV file paths and sort by starting sequence number in filename"""

        csv_files = []
        for chat_obj_folder in os.listdir(self.csv_folder):
            chat_obj_folder_path = os.path.join(self.csv_folder, chat_obj_folder)
            for csvfile in os.listdir(chat_obj_folder_path):
                if not csvfile.endswith(".csv"):
                    continue
                csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
                csv_files.append(csvfile_path)
        # Extract starting number from filename, e.g., wxid_..._0_5000.csv → 0
        pattern = re.compile(r"_(\d+)_\d+\.csv$")

        def extract_start(fp: str) -> int:
            name = os.path.basename(fp)
            m = pattern.search(name)
            return int(m.group(1)) if m else 0

        csv_files.sort(key=extract_start)
        return csv_files

    def match_qa(self, messages: List[ChatMessage]) -> List[Union[QaPair, CutMessage]]:
        """
        Match question-answer pairs

        Args:
            messages: Message list

        Returns:
            List[Union[QaPair, CutMessage]]: List of Q&A pairs containing instructions and outputs
        """
        WAITING_INSTRUCTION = "waiting_instruction"
        WAITING_RESPONSE = "waiting_response"

        current_state = WAITING_INSTRUCTION
        qa_res: List[Union[QaPair, CutMessage]] = []
        last_message = None
        current_instruction = None
        qa_id_counter = 0

        conversation_messages: List[Message] = []
        conversation_images: List[str] = []

        def _calculate_qa_length(
            messages: List[Message], new_user_content: str, new_assistant_content: str
        ) -> int:
            """Calculate total character length of messages plus new messages"""
            total_length = 0
            for msg in messages:
                total_length += len(msg.content)
            total_length += len(new_user_content) + len(new_assistant_content)
            return total_length

        def _save_current_qa_pair(
            qa_id: int,
            time_stamp: Timestamp,
            current_conversation_messages: List[Message],
            current_conversation_images: List[str],
        ) -> int:
            """Helper function to save the current QA pair."""
            nonlocal qa_res  # Allow modification of qa_res from the outer scope

            total_length = _calculate_qa_length(current_conversation_messages, "", "")

            if total_length <= self.config.messages_max_length:
                if len(current_conversation_images) > self.config.max_image_num:
                    logger.warning(
                        f"QA pair (potential id {qa_id}) with timestamp {time_stamp} "
                        f"has too many images ({len(current_conversation_images)} > {self.config.max_image_num}) "
                        "and will be skipped."
                    )
                    return qa_id

                system_content = self.system_prompt
                if self.c.add_time:
                    system_content += f" Current datetime: {time_stamp.strftime('%m-%d %H:%M:%S')}"

                qa_pair = self.QaPair(
                    id=qa_id,
                    time=time_stamp,
                    score=0,
                    messages=current_conversation_messages.copy(),
                    images=current_conversation_images.copy(),
                    system=system_content,
                )
                qa_res.append(qa_pair)
                return qa_id + 1
            else:
                logger.warning(
                    f"QA pair (potential id {qa_id}) with timestamp {time_stamp} "
                    f"exceeds max length ({total_length} > {self.config.messages_max_length}) "
                    "and will be skipped."
                )
                return qa_id

        for msg in messages:
            if isinstance(msg, CutMessage):
                # When encountering CutMessage, save current conversation and reset state
                if conversation_messages:
                    qa_id_counter = _save_current_qa_pair(
                        qa_id_counter,
                        last_message.CreateTime if last_message else msg.CreateTime,
                        conversation_messages,
                        conversation_images,
                    )
                # Reset state
                current_state = WAITING_INSTRUCTION
                current_instruction = None
                last_message = None
                conversation_messages = []
                conversation_images = []
                continue

            if current_state == WAITING_INSTRUCTION:
                if msg.is_sender == 0:  # Received message from other party
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        # If not the same conversation and there is a previous message, save the previous conversation
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,
                                conversation_messages,
                                conversation_images,
                            )
                            conversation_messages = []
                            conversation_images = []

                    # Regardless of whether a new conversation has just been started, this 'msg' now becomes the current instruction.
                    current_instruction = msg
                    last_message = msg
                    current_state = WAITING_RESPONSE

            elif current_state == WAITING_RESPONSE:
                if msg.is_sender == 0:  # Received message from other party
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,
                                conversation_messages,
                                conversation_images,
                            )
                            conversation_messages = []
                            conversation_images = []
                    current_instruction = msg
                    last_message = msg
                    # State remains unchanged
                else:  # Own message - use strategy to determine if it belongs to the same conversation
                    if last_message and self.qa_match_strategy.is_same_conversation([last_message], msg):
                        if current_instruction is None:
                            raise ValueError("current_instruction should not be None when creating a QA pair")

                        conversation_messages.append(Message(role="user", content=current_instruction.msg))
                        conversation_messages.append(Message(role="assistant", content=msg.msg))
                        if hasattr(current_instruction, "src") and current_instruction.src:
                            if isinstance(current_instruction.src, list):
                                valid_images = [img_src for img_src in current_instruction.src if img_src]
                                if valid_images:
                                    conversation_images.extend(valid_images)
                            elif current_instruction.src:
                                conversation_images.append(current_instruction.src)
                        last_message = msg

                    # Regardless of whether it matches, reset state
                    current_state = WAITING_INSTRUCTION
                    current_instruction = None

        # Process the last conversation
        if conversation_messages and last_message:
            qa_id_counter = _save_current_qa_pair(
                qa_id_counter,
                last_message.CreateTime,
                conversation_messages,
                conversation_images,
            )

        return qa_res

    def group_consecutive_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Combine multiple consecutive messages from the same person into one message, add cut when encountering cut_type

        Args:
            messages: Message list

        Returns:
            List[ChatMessage]: Combined message list
        """
        if not messages:
            return []

        def _combine_text(messages: List[ChatMessage]) -> ChatMessage:
            """
            Merge multiple messages into one

            Args:
                messages: List of messages to merge

            Returns:
                ChatMessage: Merged message
            """
            base_msg = messages[0]
            combined_content = messages[0].msg
            combined_src_list = [messages[0].src] if messages[0].modality == DataModality.IMAGE else []

            for i in messages[1:]:
                content = i.msg
                if not content:
                    continue

                if combined_content and combined_content[-1] not in [
                    "。",
                    ".",
                    "！",
                    "!",
                    "？",
                    "?",
                    "…",
                    "，",
                    ",",
                ]:
                    combined_content += "\n"

                if i.modality == DataModality.IMAGE:
                    combined_src_list.append(i.src)

                combined_content += content

            if len(combined_content) > self.c.combine_msg_max_length:
                logger.warning(
                    f"Combined message length exceeds {self.c.combine_msg_max_length}, will truncate: {combined_content[:50]}"
                )
                combined_content = combined_content[: self.c.combine_msg_max_length]
                remaining_image_count = combined_content.count("<image>")
                if len(combined_src_list) > remaining_image_count:
                    combined_src_list = combined_src_list[:remaining_image_count]

            combined_message = ChatMessage(
                id=base_msg.id,
                MsgSvrID=base_msg.MsgSvrID,
                type_name=base_msg.type_name,
                is_sender=base_msg.is_sender,
                talker=base_msg.talker,
                room_name=base_msg.room_name,
                msg=combined_content,
                src=combined_src_list,  # type: ignore
                CreateTime=messages[-1].CreateTime,  # Use the time of the last message
                modality=base_msg.modality,
                is_forward=base_msg.is_forward,
            )

            return combined_message

        def _create_cut_message(message: ChatMessage) -> CutMessage:
            return CutMessage(
                is_sender=message.is_sender,
                cut_type=message.type_name,
                CreateTime=message.CreateTime,
            )

        def _combine_current_group(group):
            """
            Process current message group and add to grouped_messages

            Args:
                group: Current message group
            """
            if len(group) > 1:
                combined_msg = _combine_text(group)
                grouped_messages.append(combined_msg)
            else:
                grouped_messages.append(group[0])

        grouped_messages = []
        current_group = []

        for _, current_msg in enumerate(messages):
            if current_msg.type_name in self.cut_type_list or (
                current_msg.modality == DataModality.IMAGE and current_msg.is_sender == 1
            ):  # Own image messages need to be cut
                if current_group:
                    # Current group has messages, combine current group and add a cut
                    _combine_current_group(current_group)
                    current_group = []

                    cut_msg = _create_cut_message(current_msg)
                    grouped_messages.append(cut_msg)
                else:
                    # Current group has no messages, check previous group
                    if grouped_messages:
                        if not isinstance(grouped_messages[-1], CutMessage):
                            cut_msg = _create_cut_message(current_msg)
                            grouped_messages.append(cut_msg)
                    # If previous group has no messages or last one is CutMessage, continue directly
                continue

            if not current_group:
                current_group = [current_msg]
                continue

            last_msg = current_group[-1]

            # Determine if it's consecutive messages from the same person
            if (
                current_msg.is_sender == last_msg.is_sender
                and current_msg.talker == last_msg.talker
                and self.single_combine_strategy.is_same_conversation([last_msg], current_msg)
            ):
                current_group.append(current_msg)
            else:
                # Not messages from the same person, process current group and start new group
                _combine_current_group(current_group)
                # Start new group
                current_group = [current_msg]

        # Process the last group of messages
        if current_group:
            _combine_current_group(current_group)

        return grouped_messages

    def process_by_msgtype(self, chat_message: ChatMessage):
        if chat_message.type_name.lower() in ["文本", "text"]:
            self.process_text(chat_message)
        # elif chat_message.modality == DataModality.IMAGE:
        #     self.process_image(chat_message)

    def load_csv(self, file_path) -> List[ChatMessage]:
        """
        Perform overall first preprocessing, filter rows that don't meet conditions, check if images exist and change type to cut if not, add DataModality field
        """
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            dtype={"msg": str, "src": str},
            escapechar=None,
            keep_default_na=False,
        )

        df = df[~df["type_name"].isin(values=self.skip_type_list)]

        if "is_forward" in df.columns:
            df = df[~((df["is_sender"] == 1) & (df["is_forward"]))]

        # Batch process text messages for PII detection and blocked words
        text_indices = []
        text_messages = []

        for i in df.index:
            if df.loc[i, "type_name"].lower() in ["文本", "text"]:  # type: ignore
                msg_str = str(df.loc[i, "msg"])
                msg_str = msg_str.replace("\n", "")
                text_indices.append(i)
                text_messages.append(msg_str)

        # TODO Deleting directly by batch_has_pii returning true/false.
        indices_to_drop = []
        if text_messages:
            pii_results = self.pii_detector.batch_has_pii(text_messages)

            for idx, (df_index, msg_str, has_pii) in enumerate(zip(text_indices, text_messages, pii_results)):
                if has_pii:
                    indices_to_drop.append(df_index)
                    continue

                # Check blocked words
                for blocked_word in self.blocked_words:
                    if blocked_word in msg_str:
                        indices_to_drop.append(df_index)
                        break

        df = df.drop(index=indices_to_drop)

        # Process other message types
        for i in df.index:
            if df.loc[i, "type_name"].lower() in ["文本", "text"]:
                continue
            if df.loc[i, "type_name"].lower() in ["图片", "image"]:  # type: ignore
                if self.c.platform in [PlatformType.WECHAT, PlatformType.TELEGRAM]:
                    result = check_image_file_exists(str(df.loc[i, "src"]))
                    if isinstance(result, str) and df.loc[i, "is_sender"] == 0:
                        df.loc[i, "src"] = result
                        df.loc[i, "msg"] = "<image>"
                        df.loc[i, "modality"] = DataModality.IMAGE
                    else:
                        df.loc[i, "type_name"] = "Cut"
            elif df.loc[i, "type_name"] in ["sticker"]:
                if self.c.platform in [PlatformType.WECHAT, PlatformType.TELEGRAM]:
                    df.loc[i, "src"] = ""
                    continue
            else:
                df.loc[i, "msg"] = ""

        df = df.dropna(how="all")
        # Time format: 2021-07-07 10:27:23
        df["CreateTime"] = pd.to_datetime(df["CreateTime"])

        return [ChatMessage(**row) for row in df.to_dict("records")]  # type: ignore

    def process_text(self, chat_message: ChatMessage):
        pass

    def save_result(self, qa_res: List[QaPair]):
        """
        Saves the list of QaPair objects to a JSON file after converting them to dictionaries.

        Args:
            qa_res: A list of QaPair objects.
        """
        processed_qa_res = []
        for idx, item in enumerate(qa_res):
            item_dict = {
                "id": idx,
                "time": item.time.isoformat() if item.time else None,
                "score": item.score,
                "messages": [{"role": msg.role, "content": msg.content} for msg in item.messages],
                "images": item.images,
                "system": item.system,
            }
            processed_qa_res.append(item_dict)

        output_path = "./dataset/res_csv/sft/sft-my.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_qa_res, f, ensure_ascii=False, indent=4)
        logger.success(
            f"Chat record processing successful, {len(qa_res)} entries in total, saved to {output_path}"
        )


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()
