import os
from pathlib import Path
import shutil
from tqdm import tqdm

from weclone.data.qa_generator import DataProcessor
from weclone.utils.log import logger

data_dir = "./dataset/wechat/dat"
wechat_data_dir = "dataset/wechat"  # 填微信个人文件夹，如C:\Users\u\Documents\WeChat Files\wxid_d6wwio22


def copy_wechat_image_dat():
    """
    根据csv里的图片路径，复制dat到指定目录
    """
    os.makedirs(data_dir, exist_ok=True)

    data_processor = DataProcessor()
    if not os.path.exists(data_processor.csv_folder) or not os.listdir(data_processor.csv_folder):
        print(f"错误：目录 '{data_processor.csv_folder}' 不存在或为空，请检查路径并确保其中包含 CSV 聊天数据文件。")
        return

    csv_files = data_processor.get_csv_files()
    print(f"共发现 {len(csv_files)} 个 CSV 文件，开始处理")
    message_list = []
    for csv_file in csv_files:
        print(f"开始处理 CSV 文件: {csv_file}")
        chat_messages = data_processor.load_csv(csv_file)
        message_list.extend(chat_messages)
        # self.process_by_msgtype(chat_message)
        print(f"处理完成: {csv_file}，共加载 {len(chat_messages)} 条消息")
    error_count = 0
    image_count = 0
    for message in tqdm(message_list):
        if message.type_name == "图片":
            # 跨平台路径处理：统一处理路径分隔符
            # 首先将所有反斜杠替换为正斜杠，然后使用pathlib处理
            image_count += 1
            normalized_src = message.src.replace("\\", "/")
            src_path = Path(normalized_src)

            # 如果是绝对路径，转换为相对路径
            if src_path.is_absolute():
                # 取路径的相对部分（去掉根目录）
                src_path = Path(*src_path.parts[1:]) if len(src_path.parts) > 1 else Path(src_path.name)

            # 构建完整路径
            image_path = Path(wechat_data_dir) / src_path
            if not image_path.exists():
                logger.warning(f"警告：路径 '{image_path}' 不存在。")
                error_count += 1
            else:
                shutil.copy(image_path, data_dir)
            # decoder.process_single_wechat_image(str(image_path), message.MsgSvrID)
    logger.info(f"复制完成，共复制 {image_count} 张图片，其中 {error_count} 张图片不存在。")


if __name__ == "__main__":
    copy_wechat_image_dat()
