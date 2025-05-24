import os
from pathlib import Path
from weclone.utils.log import logger


def check_image_file_exists(file_path: str) -> bool:
    """
    检查传入的文件路径，提取文件名（去掉前缀和扩展名），
    然后检查 dataset/images 目录下是否存在对应的文件（不论扩展名）

    Args:
        file_path: 文件路径，例如 "Storage\\Image\\2021-08\\6ce3f785b4230246639c3dd0d4a8848c.dat"

    Returns:
        bool: 如果对应的文件存在返回 True，否则返回 False

    Example:
        >>> check_image_file_exists("Storage\\Image\\2021-08\\6ce3f785b4230246639c3dd0d4a8848c.dat")
        True  # 如果 dataset/images/ 下存在任何名为 6ce3f785b4230246639c3dd0d4a8848c.* 的文件
    """
    try:
        # 手动处理路径分隔符，同时支持 Windows (\) 和 Linux (/)
        filename_with_ext = file_path.replace("\\", "/").split("/")[-1]
        # 去掉扩展名
        filename_without_ext = Path(filename_with_ext).stem

        # 使用 glob 查找任何以该文件名开头的文件
        images_dir = Path("dataset") / "images"
        matching_files = list(images_dir.glob(f"{filename_without_ext}.*"))

        exists = len(matching_files) > 0
        return exists

    except Exception as e:
        logger.error(f"检查图片文件时出错: {file_path}, 错误: {e}")
        return False


if __name__ == "__main__":
    print(check_image_file_exists("Storage\\Image\\2021-08\\6ce3f785b4230246639c3dd0d4a8848c.dat"))
