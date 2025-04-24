import os
from typing import Dict, List
import re

import pandas as pd
import json

from weclone.utils.config import load_config
from weclone.utils.log import logger
from weclone.data.models import ChatMessage, CutMessage, skip_type_list
from weclone.data.strategies import TimeWindowStrategy, LLMStrategy


class DataProcessor:
    def __init__(self):
        self.config = load_config(arg_type="make_dataset")
        self.csv_folder = "./dataset/csv"
        self.system_prompt = self.config["default_system"]
        self.cut_type_list = [
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
        ]

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

        self.c = self.config

    def main(self):
        if not os.path.exists(self.csv_folder) or not os.listdir(self.csv_folder):
            logger.error(f"错误：目录 '{self.csv_folder}' 不存在或为空，请检查路径并确保其中包含 CSV 聊天数据文件。")
            return

        csv_files = self.get_csv_files()
        message_list: List[ChatMessage] = []
        for csv_file in csv_files:
            chat_messages = self.load_csv(csv_file)
            message_list.extend(self.group_consecutive_messages(messages=chat_messages))
            # self.process_by_msgtype(chat_message)
        qa_res = self.match_qa(message_list)
        if self.c["prompt_with_history"]:
            qa_res = self.add_history_to_qa(qa_res)
        self.save_result(qa_res)

    def get_csv_files(self):
        """遍历文件夹获取所有CSV文件路径"""

        csv_files = []
        for chat_obj_folder in os.listdir(self.csv_folder):
            chat_obj_folder_path = os.path.join(self.csv_folder, chat_obj_folder)
            for csvfile in os.listdir(chat_obj_folder_path):
                if not csvfile.endswith(".csv"):
                    continue
                csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
                csv_files.append(csvfile_path)
        return csv_files

    def match_qa(self, messages: List[ChatMessage]) -> List[Dict]:
        """
        匹配问答对

        Args:
            messages: 消息列表

        Returns:
            List[Dict]: 包含指令和输出的问答对列表
        """
        # 状态定义
        WAITING_INSTRUCTION = "waiting_instruction"  # 等待指令
        WAITING_RESPONSE = "waiting_response"  # 等待回复

        current_state = WAITING_INSTRUCTION
        qa_res = []
        last_message = None
        current_instruction = None

        for msg in messages:
            # 检查是否为CutMessage
            if isinstance(msg, CutMessage):
                current_state = WAITING_INSTRUCTION
                current_instruction = None
                last_message = None
                if self.c["prompt_with_history"]:
                    qa_res.append(msg)
                continue

            if current_state == WAITING_INSTRUCTION:
                if msg.is_sender == 0:  # 收到对方消息
                    current_instruction = msg.msg
                    last_message = msg
                    current_state = WAITING_RESPONSE

            elif current_state == WAITING_RESPONSE:
                if msg.is_sender == 0:  # 收到对方消息
                    current_instruction = msg.msg
                    last_message = msg
                    # 状态保持不变
                else:  # 自己的回复 使用策略判断是否属于同一对话
                    if last_message and self.qa_match_strategy.is_same_conversation([last_message], msg):
                        qa_res.append(
                            {"instruction": current_instruction, "output": msg.msg, "system": self.system_prompt}
                        )
                    else:
                        if self.c["prompt_with_history"]:
                            qa_res.append(
                                CutMessage(
                                    is_sender=msg.is_sender,
                                    cut_type=msg.type_name,
                                    CreateTime=msg.CreateTime,
                                )
                            )
                    # 无论是否匹配，都重置状态
                    current_state = WAITING_INSTRUCTION
                    current_instruction = None
                    last_message = None

        return qa_res

    def add_history_to_qa(self, qa_res: List[Dict]) -> List[Dict]:
        qa_res_with_history = []
        last_res = {"instruction": "", "output": "", "history": [], "system": self.system_prompt}

        for _, qa in enumerate(qa_res):
            if isinstance(qa, CutMessage):
                if len(last_res["history"]) == 0:
                    continue
                else:
                    if len(last_res["history"]) == 1:
                        last_res = {
                            "system": self.system_prompt,
                            "instruction": last_res["history"][0][0],
                            "output": last_res["history"][0][1],
                            "history": [],
                        }
                    else:
                        last_res = {
                            "system": self.system_prompt,
                            "instruction": last_res["history"][-1][0],
                            "output": last_res["history"][-1][1],
                            "history": last_res["history"][:-1],
                        }
                    qa_res_with_history.append(last_res)
                    last_res = {"instruction": "", "output": "", "history": [], "system": self.system_prompt}
            else:
                last_res["history"].append([qa["instruction"], qa["output"]])

        return qa_res_with_history

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

            for i in messages[1:]:
                content = i.msg
                if not content:
                    continue

                if combined_content and combined_content[-1] not in ["。", "！", "？", "…", "，", "."]:
                    combined_content += "，"

                combined_content += content
            if len(combined_content) > self.c["combine_msg_max_length"]:
                logger.warning(f"组合后消息长度超过{self.c['combine_msg_max_length']}将截断：\n {combined_content}")
                combined_content = combined_content[: self.c["combine_msg_max_length"]]

            combined_message = ChatMessage(
                id=base_msg.id,
                MsgSvrID=base_msg.MsgSvrID,
                type_name=base_msg.type_name,
                is_sender=base_msg.is_sender,
                talker=base_msg.talker,
                room_name=base_msg.room_name,
                msg=combined_content,
                src=base_msg.src,
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
            if current_msg.type_name in self.cut_type_list:
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
        做整体第一次预处理，过滤不符合条件的行
        """
        df = pd.read_csv(file_path, encoding="utf-8", dtype={"msg": str})

        blocked_words = json.load(open("./dataset/blocked_words.json", encoding="utf-8"))["blocked_words"]

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
                for blocked_word in blocked_words:
                    if blocked_word in msg_str:
                        df = df.drop(index=i)
                        break
            else:
                df.loc[i, "msg"] = ""

        df = df.dropna(how="all")
        # 时间格式 2021-07-07 10:27:23
        # 遍历行 相同is_sender的行合并msg（）遇到不同is_sender就重新开始
        df["CreateTime"] = pd.to_datetime(df["CreateTime"])

        return [ChatMessage(*row) for row in df.values]

    def process_text(self, chat_message: ChatMessage):
        pass

    def save_result(self, qa_res: List[Dict]):
        # 保存结果
        with open(
            "./dataset/res_csv/sft/sft-my.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(qa_res, f, ensure_ascii=False)
        logger.success(f"聊天记录处理成功，共{len(qa_res)}条，保存到 {f.name}")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()
