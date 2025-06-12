import base64
import os
import time
from pathlib import Path

import requests

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

        # 使用 glob 查找精确匹配该文件名的文件（不论扩展名）
        images_dir = Path("dataset") / "media" / "images"
        matching_files = list(images_dir.glob(f"{filename_without_ext}.*"))

        if len(matching_files) > 0:
            # 获取相对于dataset/media的路径，只保留images/文件名
            full_path = matching_files[0]
            relative_path = full_path.relative_to(Path("dataset") / "media")
            return str(relative_path)
        else:
            return False

    except Exception as e:
        logger.error(f"检查图片文件时出错: {file_path}, 错误: {e}")
        return False


class ImageToTextProcessor:
    """通过兼容OpenAI API的多模态LLM将图片转换为文本。"""

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = """
        请描述这张图片的内容，重点关注：
        1. 如果是截图，描述界面内容和操作
        2. 如果是表格，描述表格结构和数据
        3. 如果是文档，提取关键文字信息
        4. 如果是生活照片，简要描述场景和内容。
        请用简洁明了的语言描述，不超过100字。"""

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"编码图片失败 {image_path}: {e}")
            return None

    def _get_image_format(self, image_path: str) -> str:
        """获取图片格式"""
        suffix = Path(image_path).suffix.lower().replace(".", "")
        if suffix == "jpg":
            return "jpeg"
        return suffix

    def _call_vision_api(self, image_path: str) -> str:
        """调用Vision API（增加了重试机制）"""
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return "[图片处理失败：无法编码]"

        image_format = self._get_image_format(image_path)

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
        }

        # --- 重试逻辑 ---
        max_retries = 5  # 最大重试次数
        base_delay = 15  # 基础等待时间（秒）

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions", headers=headers, json=payload, timeout=60
                )
                if response.status_code == 200:
                    pass
                elif response.status_code in [429, 500, 502, 503, 504]:
                    response.raise_for_status()
                else:
                    logger.error(f"API请求失败，状态码: {response.status_code}，原因: {response.reason}")
                    return "[图片描述获取失败]"

                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    logger.warning(f"API响应格式异常: {result}")
                    return "[图片描述获取失败：API格式错误]"

            except requests.exceptions.RequestException as e:
                logger.warning(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 指数退避等待
                    wait_time = base_delay * (2**attempt)
                    logger.info(f"将在 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API请求在 {max_retries} 次尝试后最终失败: {image_path}")
                    return "[图片描述获取失败：请求异常]"
            except Exception as e:
                logger.error(f"处理API响应时出现未知错误 {image_path}: {e}")
                # 对于未知错误，可以选择不重试，直接返回
                return "[图片描述获取失败：未知错误]"

        # 如果循环结束仍未成功（理论上不会执行到这里，因为上面已有返回）
        return "[图片描述获取失败：所有重试均失败]"

    def describe_image(self, image_path: str) -> str:
        """公开方法，用于描述单张图片内容"""
        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            return "[图片文件不存在]"

        logger.debug(f"正在识别图片: {os.path.basename(image_path)}")
        return self._call_vision_api(image_path)


if __name__ == "__main__":
    path = "Storage\\Image\2021-08\6ce3f785b4230246639c3dd0d4a8848c.dat"
    print(check_image_file_exists(path))
