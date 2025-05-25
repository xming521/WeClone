import os
from pathlib import Path
from weclone.utils.log import logger


def check_image_file_exists(file_path: str) -> str | bool:
    """
    检查传入的文件路径，提取文件名（去掉前缀和扩展名），
    然后检查 dataset/images 目录下是否存在对应的文件（不论扩展名）
    """
    try:
        # 直接使用 os.path.normpath 处理路径，然后转换为正斜杠
        normalized_path = os.path.normpath(file_path).replace("\\", "/")

        # 提取文件名
        filename_with_ext = os.path.basename(normalized_path)
        filename_without_ext = Path(filename_with_ext).stem

        # 使用 glob 查找任何以该文件名开头的文件
        images_dir = Path("dataset") / "images"
        matching_files = list(images_dir.glob(f"{filename_without_ext}*"))

        if len(matching_files) > 0:
            return str(matching_files[0])
        else:
            return False

    except Exception as e:
        logger.error(f"检查图片文件时出错: {file_path}, 错误: {e}")
        return False


if __name__ == "__main__":
    path = "Storage\Image\2021-08\6ce3f785b4230246639c3dd0d4a8848c.dat"
    print(check_image_file_exists(path))
