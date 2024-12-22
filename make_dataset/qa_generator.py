import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.utils.config import load_config


class DataProcessor:
    def __init__(self):
        config = load_config(arg_type="make_dataset")
        self.process_method = config["process_method"]
        self.enable_vision_model = config["enable_vision_model"]
        self.data = None
        self.processed_data = None

    def load_csv(self, file_path):
        # CSV处理逻辑
        pass

    def load_excel(self, file_path):
        # Excel处理逻辑
        pass

    def process_method1(self):
        # 处理方法1
        pass

    def process_method2(self):
        # 处理方法2
        pass

    def save_result(self, output_path):
        # 保存结果
        pass


# 使用示例
processor = DataProcessor()
processor.load_csv("input.csv")
processor.process_method1()
processor.save_result("output.csv")
