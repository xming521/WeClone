from dataclasses import dataclass
import sys
import os
from typing import List

from pandas import Timestamp
import pandas as pd
import json

current_dir = os.path.dirname(p=os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.utils.config import load_config
from make_dataset.models import ChatMessage
from make_dataset.strategies import TimeWindowStrategy, LagerModelStrategy


class DataProcessor:
    def __init__(self):
        self.config = load_config(arg_type="make_dataset")
        self.data = None
        self.processed_data = 1
        self.csv_folder = "./data/csv"
        # 根据self.config.make_dataset_args.conversation_strategy 判断初始化哪一个策略类
        if self.config["conversation_strategy"] == "time_window":
            self.conversation_strategy = TimeWindowStrategy(
                time_window=self.config["time_window"]
            )
        elif self.config["conversation_strategy"] == "lager_model":
            self.conversation_strategy = LagerModelStrategy()

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

    def main(self):
        csv_files = self.get_csv_files()
        message_list: List[ChatMessage] = []
        for csv_file in csv_files:
            chat_messages = self.load_csv(csv_file)
            # 第一次预处理后 将chat_message 加入rcsv_df_list
            message_list.append(self.group_consecutive_messages(chat_messages))
            # self.process_by_msgtype(chat_message)

    def group_consecutive_messages(
        self, messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """
        将同一个人连续发送的多条消息组合成一条消息

        Args:
            messages: 消息列表

        Returns:
            List[ChatMessage]: 组合后的消息列表
        """
        if not messages:
            return []

        grouped_messages = []
        current_group = [messages[0]]

        for i in range(1, len(messages)):
            current_msg = messages[i]
            last_msg = current_group[-1]

            # 判断是否是同一个人的连续消息
            if (
                current_msg.is_sender == last_msg.is_sender
                and current_msg.talker == last_msg.talker
                and (current_msg.CreateTime - last_msg.CreateTime).total_seconds()
                < self.config.get("message_time_window", 3600)
            ):

                # 同一个人的连续消息，添加到当前组
                current_group.append(current_msg)
            else:
                # 不是同一个人的消息，处理当前组并开始新组
                if len(current_group) > 1:
                    # 合并消息内容
                    combined_msg = self._combine_messages(current_group)
                    grouped_messages.append(combined_msg)
                else:
                    # 只有一条消息，直接添加
                    grouped_messages.append(current_group[0])

                # 开始新组
                current_group = [current_msg]

        # 处理最后一组消息
        if current_group:
            if len(current_group) > 1:
                combined_msg = self._combine_messages(current_group)
                grouped_messages.append(combined_msg)
            else:
                grouped_messages.append(current_group[0])

        return grouped_messages

    def _combine_messages(self, messages: List[ChatMessage]) -> ChatMessage:
        """
        合并多条消息为一条

        Args:
            messages: 要合并的消息列表

        Returns:
            ChatMessage: 合并后的消息
        """
        # 以第一条消息为基础
        base_msg = messages[0]

        # 合并消息内容
        combined_content = ""
        for i, msg in enumerate(messages):
            content = msg.msg.strip()
            if not content:
                continue

            if i > 0:
                # 如果前一条消息没有以标点符号结尾，添加一个句号
                if combined_content and combined_content[-1] not in [
                    "。",
                    "！",
                    "？",
                    "…",
                    "，",
                    ".",
                ]:
                    combined_content += "，"

            combined_content += content

        # 确保最后一条消息以句号结尾
        if combined_content and combined_content[-1] not in [
            "。",
            "！",
            "？",
            "…",
            ".",
        ]:
            combined_content += "。"

        # 创建新的合并消息
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

    def create_conversation_data(self, messages: List[ChatMessage]) -> dict:
        """
        将一组消息组成一条对话数据

        Args:
            messages: 属于同一对话的消息列表

        Returns:
            dict: 包含对话历史的数据字典
        """
        conversation = []
        for msg in messages:
            if msg.type_name in self.config["include_type"]:
                conversation.append(
                    {
                        "role": "user" if msg.is_sender == 0 else "assistant",
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    }
                )

        # 确保对话是按时间顺序排列的
        conversation.sort(key=lambda x: x["timestamp"])

        # 限制历史长度
        if len(conversation) > self.config["history_length"]:
            conversation = conversation[-self.config["history_length"] :]

        return {
            "conversation": conversation,
            "metadata": {
                "conversation_id": messages[0].conversation_id if messages else None,
                "timestamp": messages[-1].timestamp if messages else None,
            },
        }

    def process_by_msgtype(self, chat_message: ChatMessage):
        if chat_message.type_name == "文本":
            self.process_text(chat_message)
        # elif chat_message.type_name == "图片":
        #     self.process_image(chat_message)

    def process_by_msgtype(self, chat_message: ChatMessage):
        if chat_message.type_name == "文本":
            self.process_text(chat_message)
        # elif chat_message.type_name == "图片":
        #     self.process_image(chat_message)

    def load_csv(self, file_path) -> List[ChatMessage]:
        """
        做整体第一次预处理，删除不符合条件的行
        """
        chat_df = pd.read_csv(file_path)

        blocked_words = json.load(
            open("./make_dataset/blocked_words.json", encoding="utf-8")
        )["blocked_words"]

        type_list = [
            "文本",
            "图片",
            "卡片式链接",
            "合并转发的聊天记录",
            "视频",
            "语言",
            "未知",
            "分享的小程序",
        ]
        # chat_df = chat_df[chat_df["type_name"].isin(values=type_list)]

        # chat_df["content"] = chat_df["msg"]

        # 如果type_name为文本 并且msg 包含 手机号、身份证号、邮箱、网址则删除这行
        for i in chat_df.index:
            if chat_df.loc[i, "type_name"] == "文本":
                if (
                    "1\d{10}" in chat_df.loc[i, "msg"]
                    or "\d{18}" in chat_df.loc[i, "msg"]
                    or "\w+@\w+" in chat_df.loc[i, "msg"]
                    or "http" in chat_df.loc[i, "msg"]
                    or r"\\xa0" in chat_df.loc[i, "msg"]
                    or r"\\u" in chat_df.loc[i, "msg"]
                ):
                    chat_df = chat_df.drop(index=i)
                    continue
                for blocked_word in blocked_words:
                    if blocked_word in chat_df.loc[i, "msg"]:
                        chat_df = chat_df.drop(index=i)
                        break
            else:
                chat_df.loc[i, "msg"] = ""

        chat_df = chat_df.dropna(how="all")
        # 时间格式 2021-07-07 10:27:23
        # 遍历行 相同is_sender的行合并msg（）遇到不同is_sender就重新开始
        chat_df["CreateTime"] = pd.to_datetime(chat_df["CreateTime"])

        return [ChatMessage(*row) for row in chat_df.values]

    def process_text(self, chat_message: ChatMessage):

        pass

    def process_image(self):
        # 处理方法1
        pass

    def process_method2(self):
        # 处理方法2
        pass

    def save_result(self):
        # 保存结果
        pass


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()
