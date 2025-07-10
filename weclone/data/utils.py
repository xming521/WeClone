import base64
import concurrent.futures
import os
from pathlib import Path

import requests

from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger
from weclone.utils.retry import retry_on_http_error


def check_image_file_exists(file_path: str) -> str | bool:
    try:
        normalized_path = os.path.normpath(file_path).replace("\\", "/")

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

    def __init__(self, api_url: str, api_key: str, model_name: str, config: WCMakeDatasetConfig):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.config = config
        self.prompt = """
        请描述这张图片的内容，重点关注：
        1. 如果是截图，描述界面内容和操作
        2. 如果是表格，描述表格结构和数据
        3. 如果是文档，提取关键文字信息
        4. 如果是生活照片，简要描述场景和内容。
        请用简洁明了的语言描述，不超过100字。"""

    def _process_images_in_parallel(self, qa_list):
        """并行处理所有对话中的图片，并将描述替换回对话文本。"""
        all_image_paths = []
        media_dir = self.config.media_dir

        # 遍历所有对话，收集并构造完整的图片路径
        for qa_pair in qa_list:
            if qa_pair.images:
                image_list = qa_pair.images if isinstance(qa_pair.images, list) else [qa_pair.images]
                for relative_path in image_list:
                    full_path = os.path.join(media_dir, relative_path)
                    all_image_paths.append(full_path)

        if not all_image_paths:
            logger.info("未在对话中找到任何图片，跳过识别。")
            return qa_list

        logger.info(f"共找到 {len(all_image_paths)} 张有效图片需要识别。")
        max_workers = self.config.vision_api.max_workers

        # 使用线程池并行调用API，executor.map 会保持结果顺序与输入一致
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 现在传递给 image_processor 的是完整的路径
            image_descriptions = list(executor.map(self.describe_image, all_image_paths))

        desc_iterator = iter(image_descriptions)
        for qa_pair in qa_list:
            if not qa_pair.images:
                continue

            for message in qa_pair.messages:
                # 替换消息内容中的 <image> 占位符
                num_images_in_message = message.content.count("<image>")
                for _ in range(num_images_in_message):
                    try:
                        description = next(desc_iterator)
                        # 使用 count=1 确保每次只替换一个占位符，并添加换行符以增强可读性
                        message.content = message.content.replace(
                            "<image>", f"\n[图片描述: {description}]\n", 1
                        )
                    except StopIteration:
                        logger.error("图片数量与描述数量不匹配，可能存在逻辑错误。")
                        message.content = message.content.replace("<image>", "\n[图片描述缺失]\n", 1)

            # 清空图片列表，因为它们已被转换为文本
            qa_pair.images.clear()

        return qa_list

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"编码图片失败 {image_path}: {e}")
            return ""

    def _get_image_format(self, image_path: str) -> str:
        """获取图片格式"""
        suffix = Path(image_path).suffix.lower().replace(".", "")
        if suffix == "jpg":
            return "jpeg"
        return suffix

    @retry_on_http_error(
        max_retries=5,
        base_delay=15.0,
        max_delay=300.0,
        backoff_factor=2.0,
        retry_on_status=[429, 500, 502, 503, 504],
        retry_on_exceptions=[requests.exceptions.RequestException, ConnectionError, TimeoutError],
    )
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

        response = requests.post(
            f"{self.api_url}/chat/completions", headers=headers, json=payload, timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content.strip()
            else:
                logger.warning(f"API响应格式异常: {result}")
                return "[图片描述获取失败：API格式错误]"
        else:
            logger.error(f"API请求失败，状态码: {response.status_code}，原因: {response.reason}")
            response.raise_for_status()  # 触发重试机制
            return "[图片描述获取失败]"

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
