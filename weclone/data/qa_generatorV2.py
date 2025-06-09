import os
import sys
import subprocess
from typing import Dict, List, Union
import re

import pandas as pd
import json
from pandas import Timestamp

from weclone.data.clean.strategies import LLMCleaningStrategy
from weclone.data.clean.strategies_online import OlineLLMCleaningStrategy
from weclone.utils.config import load_config
from weclone.utils.log import logger
from weclone.data.models import (
    ChatMessage,
    CutMessage,
    skip_type_list,
    cut_type_list,
    QaPairV2,
    Message,
)
from weclone.data.strategies import TimeWindowStrategy, LLMStrategy
from weclone.data.utils import check_image_file_exists


class DataProcessor:
    def __init__(self):
        self.config = load_config(arg_type="make_dataset")
        self.csv_folder = "./dataset/csv"
        self.system_prompt = self.config["default_system"]

        # msg_type
        logger.info("image 类型消息,使用sharegpt格式")
        self.QaPair = QaPairV2
        self.config["prompt_with_history"] = True

        self.include_type = self.config.get("include_type", [])
        if self.config["platform"] == "wechat":
            self.cut_type_list = cut_type_list.get_items(lang="zh_CN")
            self.include_type = cut_type_list.translate_batch(texts=[t for t in self.include_type if t != "text"])
            self.cut_type_list = [t for t in self.cut_type_list if t not in self.include_type]
        else:
            self.cut_type_list = cut_type_list.get_items(lang="en")

        # blocked_words
        config_blocked_words = self.config.get("blocked_words", [])
        file_blocked_words = []
        try:
            with open("./dataset/blocked_words.json", encoding="utf-8") as f:
                file_blocked_words = json.load(f).get("blocked_words", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        self.blocked_words = list(set(config_blocked_words + file_blocked_words))
        # logger.info(f"聊天记录禁用词: {self.blocked_words}")

        # combine_strategy
        if self.config["single_combine_strategy"] == "time_window":
            self.single_combine_strategy = TimeWindowStrategy(
                time_window=self.config["single_combine_time_window"] * 60,
                is_single_chat=True,
            )
        elif self.config["single_combine_strategy"] == "llm":
            self.single_combine_strategy = LLMStrategy(
                is_single_chat=True,
            )

        if self.config["qa_match_strategy"] == "time_window":
            self.qa_match_strategy = TimeWindowStrategy(
                time_window=self.config["qa_match_time_window"] * 60,
                is_single_chat=False,
            )
        elif self.config["qa_match_strategy"] == "llm":
            self.qa_match_strategy = LLMStrategy(is_single_chat=False)

        # clean_dataset
        clean_dataset_config = self.config.get("clean_dataset", {})
        enable_clean = clean_dataset_config.get("enable_clean", False)

        if enable_clean:
            if self.config.get("prompt_with_history", False):
                logger.error("开启 prompt_with_history 不支持 clean_dataset 功能")
                exit()

            if "image" in self.config.get("include_type", []):
                logger.error("开启 clean_dataset 不支持 image 类型消息")
                exit()

            from llamafactory.extras.packages import is_vllm_available

            if not is_vllm_available():
                logger.warning("vLLM 不可用，暂不清洗数据集。")
                clean_dataset_config["enable_clean"] = False

        if self.config.get("clean_dataset", {}).get("enable_clean", False):
            if self.config.get("clean_dataset", {}).get("clean_strategy", "llm") == "llm":
                if self.config.get("online_llm_clear"):
                    self.clean_strategy = OlineLLMCleaningStrategy(make_dataset_config=self.config)
                else:
                    self.clean_strategy = LLMCleaningStrategy(make_dataset_config=self.config)
        self.c = self.config

    def main(self):
        if not os.path.exists(self.csv_folder) or not os.listdir(self.csv_folder):
            logger.error(f"错误：目录 '{self.csv_folder}' 不存在或为空，请检查路径并确保其中包含 CSV 聊天数据文件。")
            return

        csv_files = self.get_csv_files()
        logger.info(f"共发现 {len(csv_files)} 个 CSV 文件,开始处理,请耐心等待...")
        message_list: List[ChatMessage] = []
        for csv_file in csv_files:
            logger.debug(f"开始处理 CSV 文件: {csv_file}")
            chat_messages = self.load_csv(csv_file)
            message_list.extend(self.group_consecutive_messages(messages=chat_messages))
            # self.process_by_msgtype(chat_message)
            logger.debug(f"处理完成: {csv_file}，共加载 {len(chat_messages)} 条消息")
        qa_res = self.match_qa(message_list)
        qa_res = [item for item in qa_res if isinstance(item, QaPairV2)]

        if self.c.get("clean_dataset", {}).get("enable_clean", False):
            self.clean_strategy.judge(qa_res)  # type: ignore
            # qa_res = self.clean_strategy.clean(qa_res) #改到sft.py中
        self.save_result(qa_res)
        self._execute_length_cdf_script()

        logger.success(f"聊天记录处理成功，共{len(qa_res)}条，保存到 ./dataset/res_csv/sft/sft-my.json")

    def _execute_length_cdf_script(self):
        """执行 length_cdf.py 脚本来计算cutoff_len。"""
        try:
            python_executable = sys.executable
            # 脚本路径是相对于项目根目录的
            script_path = os.path.join("weclone", "utils", "length_cdf.py")

            command_parts = [
                python_executable,
                script_path,
                f'--model_name_or_path="{self.c["model_name_or_path"]}"',
                f'--dataset="{self.c["dataset"]}"',
                f'--dataset_dir="{self.c["dataset_dir"]}"',
                f'--template="{self.c["template"]}"',
                "--interval=512",
            ]

            if "media_dir" in self.c:
                command_parts.append(f'--media_dir="{self.c["media_dir"]}"')
            if "image_max_pixels" in self.c:
                command_parts.append(f'--image_max_pixels="{self.c["image_max_pixels"]}"')

            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = "0"
            child_env["LLAMAFACTORY_VERBOSITY"] = "ERROR"

            process = subprocess.Popen(
                command_parts,
                env=child_env,
                stdout=None,  # 使用 None 表示使用父进程的标准输出（即终端）
                stderr=None,  # 使用 None 表示使用父进程的标准错误（即终端）
                text=True,
                bufsize=1,  # 行缓冲
            )
            return_code = process.wait()
            if return_code != 0:
                logger.error(f"命令 '{' '.join(command_parts)}' 执行失败，返回码 {return_code}")
        except FileNotFoundError:
            # command_parts[0] 是 python_executable, command_parts[1] 是 script_path
            logger.error(f"命令执行失败: 找不到可执行文件 '{command_parts[0]}' 或脚本 '{command_parts[1]}'")
        except KeyError as e:
            logger.error(f"执行 length_cdf.py 脚本失败：配置项缺失 {str(e)}")
        except Exception as e:
            logger.error(f"执行 length_cdf.py 脚本时发生未知错误: {str(e)}")

    def get_csv_files(self):
        """遍历文件夹获取所有CSV文件路径，并按文件名中的起始序号排序"""

        csv_files = []
        for chat_obj_folder in os.listdir(self.csv_folder):
            chat_obj_folder_path = os.path.join(self.csv_folder, chat_obj_folder)
            for csvfile in os.listdir(chat_obj_folder_path):
                if not csvfile.endswith(".csv"):
                    continue
                csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
                csv_files.append(csvfile_path)
        # 提取文件名中的起始数字，比如 wxid_..._0_5000.csv → 0
        pattern = re.compile(r"_(\d+)_\d+\.csv$")

        def extract_start(fp: str) -> int:
            name = os.path.basename(fp)
            m = pattern.search(name)
            return int(m.group(1)) if m else 0

        # 按起始数字升序排序
        csv_files.sort(key=extract_start)
        return csv_files

    def match_qa(self, messages: List[ChatMessage]) -> List[Union[QaPairV2, CutMessage]]:
        """
        匹配问答对，直接处理历史对话

        Args:
            messages: 消息列表

        Returns:
            List[Union[QaPairV2, CutMessage]]: 包含指令和输出的问答对列表
        """
        # 状态定义
        WAITING_INSTRUCTION = "waiting_instruction"  # 等待指令
        WAITING_RESPONSE = "waiting_response"  # 等待回复

        current_state = WAITING_INSTRUCTION
        qa_res: List[Union[QaPairV2, CutMessage]] = []
        last_message = None
        current_instruction = None
        qa_id_counter = 0

        # 用于构建历史对话的变量
        conversation_messages: List[Message] = []
        conversation_images: List[str] = []

        def _calculate_qa_length(messages: List[Message], new_user_content: str, new_assistant_content: str) -> int:
            """计算messages加上新消息后的总字符长度"""
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

            if total_length <= self.config.get("messages_max_length", 4096):
                if len(current_conversation_images) > self.config.get("max_image_num", 2):
                    logger.warning(
                        f"QA pair (potential id {qa_id}) with timestamp {time_stamp} "
                        f"has too many images ({len(current_conversation_images)} > {self.config.get('max_image_num', 2)}) "
                        "and will be skipped."
                    )
                    return qa_id

                qa_pair = self.QaPair(
                    id=qa_id,
                    time=time_stamp,
                    score=0,
                    messages=current_conversation_messages.copy(),
                    images=current_conversation_images.copy(),
                    system=self.system_prompt,
                )
                qa_res.append(qa_pair)
                return qa_id + 1
            else:
                logger.warning(
                    f"QA pair (potential id {qa_id}) with timestamp {time_stamp} "
                    f"exceeds max length ({total_length} > {self.config.get('messages_max_length', 4096)}) "
                    "and will be skipped."
                )
                return qa_id

        for msg in messages:
            if isinstance(msg, CutMessage):
                # 遇到 CutMessage，保存当前对话并重置状态
                if conversation_messages:
                    qa_id_counter = _save_current_qa_pair(
                        qa_id_counter,
                        last_message.CreateTime if last_message else msg.CreateTime,
                        conversation_messages,
                        conversation_images,
                    )
                # 重置状态
                current_state = WAITING_INSTRUCTION
                current_instruction = None
                last_message = None
                conversation_messages = []
                conversation_images = []
                continue

            if current_state == WAITING_INSTRUCTION:
                if msg.is_sender == 0:  # 收到对方消息 (potential instruction)
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        # 如果不是同一段对话，且存在上一条消息，则保存之前的对话
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,  # 使用上一条消息的时间
                                conversation_messages,
                                conversation_images,
                            )
                            conversation_messages = []
                            conversation_images = []

                    # 无论是否刚刚重新开启了一段对话，这个 'msg' 现在都成为当前的指令。
                    current_instruction = msg
                    last_message = msg
                    current_state = WAITING_RESPONSE

            elif current_state == WAITING_RESPONSE:
                if msg.is_sender == 0:  # 收到对方消息
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        # 如果不是同一段对话，且存在上一条消息，则保存之前的对话
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,  # 使用上一条消息的时间
                                conversation_messages,
                                conversation_images,
                            )
                            conversation_messages = []
                            conversation_images = []
                    current_instruction = msg
                    last_message = msg
                    # 状态保持不变
                else:  # 自己的回复 使用策略判断是否属于同一对话
                    if last_message and self.qa_match_strategy.is_same_conversation([last_message], msg):
                        assert current_instruction is not None, (
                            "current_instruction should not be None when creating a QA pair"
                        )

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

                    # 无论是否匹配，都重置状态
                    current_state = WAITING_INSTRUCTION
                    current_instruction = None

        # 处理最后的对话
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
        将同一个人连续发送的多条消息组合成一条消息，遇到cut_type添加cut

        Args:
            messages: 消息列表

        Returns:
            List[ChatMessage]: 组合后的消息列表
        """
        if not messages:
            return []

        def _combine_text(messages: List[ChatMessage]) -> ChatMessage:
            """
            合并多条消息为一条

            Args:
                messages: 要合并的消息列表

            Returns:
                ChatMessage: 合并后的消息
            """
            base_msg = messages[0]
            combined_content = messages[0].msg
            combined_src_list = [messages[0].src] if messages[0].type_name in ["图片", "image"] else []

            for i in messages[1:]:
                content = i.msg
                if not content:
                    continue

                if combined_content and combined_content[-1] not in [
                    "。",
                    "！",
                    "？",
                    "…",
                    "，",
                    ".",
                ]:
                    combined_content += "，"

                if i.type_name == "图片":
                    # combined_content += "<image>"
                    combined_src_list.append(i.src)

                combined_content += content

            if len(combined_content) > self.c["combine_msg_max_length"]:
                # TODO: 可能会截断<image>
                logger.warning(
                    f"组合后消息长度超过{self.c['combine_msg_max_length']}将截断：\n {combined_content[:50]}"
                )
                combined_content = combined_content[: self.c["combine_msg_max_length"]]

            combined_message = ChatMessage(
                id=base_msg.id,
                MsgSvrID=base_msg.MsgSvrID,
                type_name=base_msg.type_name,
                is_sender=base_msg.is_sender,
                talker=base_msg.talker,
                room_name=base_msg.room_name,
                msg=combined_content,
                src=combined_src_list,  # type: ignore
                CreateTime=messages[-1].CreateTime,  # 使用最后一条消息的时间
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
            处理当前消息组并添加到grouped_messages

            Args:
                group: 当前消息组
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
                current_msg.type_name in ["图片", "image"] and current_msg.is_sender == 1
            ):  # 自己发图要cut
                if current_group:
                    # 当前组有消息，合并当前组，并添加一条cut
                    _combine_current_group(current_group)
                    current_group = []

                    cut_msg = _create_cut_message(current_msg)
                    grouped_messages.append(cut_msg)
                else:
                    # 当前组没消息，检查上一个组
                    if grouped_messages:
                        if not isinstance(grouped_messages[-1], CutMessage):
                            cut_msg = _create_cut_message(current_msg)
                            grouped_messages.append(cut_msg)
                    # 如果上一个组没消息或最后一条是CutMessage，直接continue
                continue

            if not current_group:
                current_group = [current_msg]
                continue

            last_msg = current_group[-1]

            # 判断是否是同一个人的连续消息
            if (
                current_msg.is_sender == last_msg.is_sender
                and current_msg.talker == last_msg.talker
                and self.single_combine_strategy.is_same_conversation([last_msg], current_msg)
            ):
                current_group.append(current_msg)
            else:
                # 不是同一个人的消息，处理当前组并开始新组
                _combine_current_group(current_group)
                # 开始新组
                current_group = [current_msg]

        # 处理最后一组消息
        if current_group:
            _combine_current_group(current_group)

        return grouped_messages

    def process_by_msgtype(self, chat_message: ChatMessage):
        if chat_message.type_name == "文本":
            self.process_text(chat_message)
        # elif chat_message.type_name == "图片":
        #     self.process_image(chat_message)

    def load_csv(self, file_path) -> List[ChatMessage]:
        """
        做整体第一次预处理，过滤不符合条件的行，检查图片是否存不存在类型改为cut
        """
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            dtype={"msg": str, "src": str},
            escapechar=None,
        )

        df["src"] = df["src"].fillna("")
        df = df[~df["type_name"].isin(values=skip_type_list)]

        # 如果type_name为文本 并且msg 包含 手机号、身份证号、邮箱、网址则删除这行
        for i in df.index:
            if df.loc[i, "type_name"] == "文本":
                msg_str = str(df.loc[i, "msg"])
                if (
                    re.search(r"1\d{10}", msg_str)
                    or re.search(r"\d{18}", msg_str)
                    or re.search(r"\w+@\w+", msg_str)
                    or "http" in msg_str
                    or r"\\xa0" in msg_str
                    or r"\\u" in msg_str
                ):
                    df = df.drop(index=i)
                    continue
                for blocked_word in self.blocked_words:
                    if blocked_word in msg_str:
                        df = df.drop(index=i)
                        break
            elif df.loc[i, "type_name"] == "图片":
                if self.c["platform"] == "wechat":
                    result = check_image_file_exists(str(df.loc[i, "src"]))
                    if isinstance(result, str):
                        df.loc[i, "src"] = result
                        df.loc[i, "msg"] = "<image>"
                    else:
                        df.loc[i, "type_name"] = "Cut"

            else:
                df.loc[i, "msg"] = ""

        df = df.dropna(how="all")
        # 时间格式 2021-07-07 10:27:23
        # 遍历行 相同is_sender的行合并msg（）遇到不同is_sender就重新开始
        df["CreateTime"] = pd.to_datetime(df["CreateTime"])

        return [ChatMessage(*row) for row in df.values]

    def process_text(self, chat_message: ChatMessage):
        pass

    def save_result(self, qa_res: List[QaPairV2]):
        """
        Saves the list of QaPairV2 objects to a JSON file after converting them to dictionaries.

        Args:
            qa_res: A list of QaPairV2 objects.
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
        logger.success(f"聊天记录处理成功，共{len(qa_res)}条，保存到 {output_path}")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()
