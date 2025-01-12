from dataclasses import dataclass
import sys
import os
from typing import List

from pandas import Timestamp
import pandas as pd
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.utils.config import load_config


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


class DataProcessor:
    def __init__(self):
        self.config = load_config(arg_type="make_dataset")
        self.data = None
        self.processed_data = 1
        self.csv_folder = "./data/csv"

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
        for csv_file in csv_files:
            chat_messages = self.load_csv(csv_file)
            print(chat_messages)

    def load_csv(self, file_path) -> List[ChatMessage]:
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

    def load_excel(self, file_path):
        # Excel处理逻辑
        pass

    def process_image(self):
        # 处理方法1
        pass

    def process_method2(self):
        # 处理方法2
        pass

    def save_result(self, output_path):
        # 保存结果
        pass


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()
