import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from weclone.data.models import ChatMessage
from weclone.data.qa_generator import DataProcessor
from weclone.utils.log import logger

data_dir = "./dataset/wechat/dat"


def copy_wechat_image_dat(wechat_data_dir):
    """
    根据csv里的图片路径，复制dat到指定目录

    Args:
        wechat_data_dir (str): 微信个人文件夹路径
    """
    os.makedirs(data_dir, exist_ok=True)

    data_processor = DataProcessor()
    if not os.path.exists(data_processor.csv_folder) or not os.listdir(data_processor.csv_folder):
        print(
            f"错误：目录 '{data_processor.csv_folder}' 不存在或为空，请检查路径并确保其中包含 CSV 聊天数据文件。"
        )
        return

    csv_files = data_processor.get_csv_files()
    print(f"共发现 {len(csv_files)} 个 CSV 文件，开始处理")
    message_list = []
    for csv_file in csv_files:
        print(f"开始处理 CSV 文件: {csv_file}")
        df = pd.read_csv(csv_file, encoding="utf-8", dtype={"msg": str}, escapechar=None)
        message_list.extend([ChatMessage(*row) for row in df.values])
        print(f"处理完成: {csv_file}，共加载 {len(message_list)} 条消息")
    error_count = 0
    image_count = 0
    for message in tqdm(message_list):
        if message.type_name == "图片" and message.is_sender == 0:  # 只要对方发送的图片
            # 跨平台路径处理：统一处理路径分隔符
            # 首先将所有反斜杠替换为正斜杠，然后使用pathlib处理
            image_count += 1
            normalized_src = message.src.replace("\\", "/")
            src_path = Path(normalized_src)

            # 如果是绝对路径，转换为相对路径
            if src_path.is_absolute():
                # 取路径的相对部分（去掉根目录）
                src_path = Path(*src_path.parts[1:]) if len(src_path.parts) > 1 else Path(src_path.name)

            # 将文件扩展名改为.dat
            src_path = src_path.with_suffix(".dat")

            # 构建完整路径
            image_path = Path(wechat_data_dir) / src_path
            if not image_path.exists():
                logger.warning(f"警告：路径 '{image_path}' 不存在。")
                error_count += 1
            else:
                shutil.copy(image_path, data_dir)
            # decoder.process_single_wechat_image(str(image_path), message.MsgSvrID)
    logger.info(f"复制完成，共复制 {image_count} 张图片，其中 {error_count} 张图片不存在。")


def main():
    parser = argparse.ArgumentParser(description="微信聊天记录图片处理工具")
    parser.add_argument(
        "--wechat-data-dir",
        type=str,
        required=True,
        help=r"微信个人文件夹路径，例如: C:\Users\user\Documents\WeChat Files\wxid_d645wirus6mo22",
    )

    args = parser.parse_args()

    wechat_data_dir = os.path.normpath(args.wechat_data_dir)

    logger.info(f"使用微信数据目录: {wechat_data_dir}")
    copy_wechat_image_dat(wechat_data_dir)


if __name__ == "__main__":
    main()
