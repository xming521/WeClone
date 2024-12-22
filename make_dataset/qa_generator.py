from dataclasses import dataclass
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.utils.config import load_config


@dataclass
class ChatMessage:
    id: int
    MsgSvrID: str
    type_name: str
    is_sender: int
    talker: str
    room_name: str
    msg: str
    src: str
    CreateTime: str


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

    def process(self):
        csv_files = self.get_csv_files()

    def load_csv(self, file_path):
        # CSV处理逻辑
        pass

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
    processor.load_csv("input.csv")
    processor.save_result("output.csv")
