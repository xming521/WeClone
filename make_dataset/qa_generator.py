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
from make_dataset.models import ChatMessage, CutMessage, skip_type_list
from make_dataset.strategies import TimeWindowStrategy, LagerModelStrategy


class DataProcessor:
    def __init__(self):
        self.config = load_config(arg_type="make_dataset")
        self.data = None
        self.csv_folder = "./data/csv"
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
        ]

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
            message_list.append(self.group_consecutive_messages(messages=chat_messages))
            # self.process_by_msgtype(chat_message)

    def group_consecutive_messages(
        self, messages: List[ChatMessage]
    ) -> List[ChatMessage]:
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
            combined_content = messages[0].msg.strip()

            for i in messages[1:]:
                content = i.msg.strip()
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

                combined_content += content

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
            """
            创建一个CutMessage实例

            Args:
                message: 当前处理的消息，用于获取属性

            Returns:
                CutMessage: 创建的CutMessage实例
            """
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
                and (current_msg.CreateTime - last_msg.CreateTime).total_seconds()
                < 3600
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
                        "content": msg.msg,
                        "timestamp": msg.CreateTime,
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
                "conversation_id": str(messages[0].id) if messages else None,
                "timestamp": messages[-1].CreateTime if messages else None,
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
        做整体第一次预处理，过滤不符合条件的行
        """
        df = pd.read_csv(file_path, encoding="utf-8", dtype={"msg": str})

        blocked_words = json.load(
            open("./make_dataset/blocked_words.json", encoding="utf-8")
        )["blocked_words"]

        df = df[~df["type_name"].isin(values=skip_type_list)]

        # 如果type_name为文本 并且msg 包含 手机号、身份证号、邮箱、网址则删除这行
        for i in df.index:
            if df.loc[i, "type_name"] == "文本":
                if (
                    "1\d{10}" in df.loc[i, "msg"]
                    or "\d{18}" in df.loc[i, "msg"]
                    or "\w+@\w+" in df.loc[i, "msg"]
                    or "http" in df.loc[i, "msg"]
                    or r"\\xa0" in df.loc[i, "msg"]
                    or r"\\u" in df.loc[i, "msg"]
                ):
                    df = df.drop(index=i)
                    continue
                for blocked_word in blocked_words:
                    if blocked_word in df.loc[i, "msg"]:
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
