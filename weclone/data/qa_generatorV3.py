import os
import sys
import subprocess
from typing import List, Union
import re
import json
import base64
import requests
from pathlib import Path
import concurrent.futures

from weclone.utils.log import logger
from weclone.data.models import QaPairV2, Message, ChatMessage

from weclone.data.qa_generatorV2 import DataProcessor as BaseDataProcessor


class ImageToTextProcessor:
    """通过兼容OpenAI API的多模态LLM将图片转换为文本。"""
    def __init__(self, api_url: str, api_key: str, model_name: str, prompt: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = prompt

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图片失败 {image_path}: {e}")
            return None

    def _get_image_format(self, image_path: str) -> str:
        """获取图片格式"""
        suffix = Path(image_path).suffix.lower().replace('.', '')
        if suffix == 'jpg':
            return 'jpeg'
        return suffix

    def _call_vision_api(self, image_path: str) -> str:
        """调用Vision API"""
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return "[图片处理失败：无法编码]"

        image_format = self._get_image_format(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                logger.warning(f"API响应格式异常: {result}")
                return "[图片描述获取失败：API格式错误]"
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败 {image_path}: {e}")
            return "[图片描述获取失败：请求异常]"
        except Exception as e:
            logger.error(f"处理API响应时出错 {image_path}: {e}")
            return "[图片描述获取失败：未知错误]"

    def describe_image(self, image_path: str) -> str:
        """公开方法，用于描述单张图片内容"""
        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            return "[图片文件不存在]"
        
        logger.debug(f"正在识别图片: {os.path.basename(image_path)}")
        return self._call_vision_api(image_path)


class DataProcessor(BaseDataProcessor):
    """
    继承自v2的DataProcessor，并增加了图片识别处理功能。
    """
    def __init__(self):
        super().__init__()
        
        # 基于配置初始化图片识别处理器
        vision_config = self.config.get("vision_api", {})
        if vision_config.get("enable", False) and vision_config.get("api_key"):
            self.image_processor = ImageToTextProcessor(
                api_url=vision_config.get("api_url", "https://api.openai.com/v1"),
                api_key=vision_config.get("api_key"),
                model_name=vision_config.get("model_name", "gpt-4o"),
                prompt=vision_config.get("prompt", "请详细描述这张图片的内容。")
            )
            logger.info(f"已启用图片识别功能, 模型: {self.image_processor.model_name}")
        else:
            self.image_processor = None

    def _process_images_in_parallel(self, qa_list: List[QaPairV2]) -> List[QaPairV2]:
        """并行处理所有对话中的图片，并将描述替换回对话文本。"""
        all_image_paths = []
        media_dir = self.c.get("media_dir", "dataset/media")

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
        max_workers = self.c.get("vision_api", {}).get("max_workers", 8)
        
        # 使用线程池并行调用API，executor.map 会保持结果顺序与输入一致
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 现在传递给 image_processor 的是完整的路径
            image_descriptions = list(executor.map(self.image_processor.describe_image, all_image_paths))

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
                        message.content = message.content.replace("<image>", f"\n[图片描述: {description}]\n", 1)
                    except StopIteration:
                        logger.error("图片数量与描述数量不匹配，可能存在逻辑错误。")
                        message.content = message.content.replace("<image>", "\n[图片描述缺失]\n", 1)
            
            # 清空图片列表，因为它们已被转换为文本
            qa_pair.images.clear()

        return qa_list

    def main(self):
        """
        重写 main 方法以集成图片处理流程。
        """
        if not os.path.exists(self.csv_folder) or not os.listdir(self.csv_folder):
            logger.error(
                f"错误：目录 '{self.csv_folder}' 不存在或为空，请检查路径并确保其中包含 CSV 聊天数据文件。"
            )
            return

        # 调用继承自父类的方法
        csv_files = self.get_csv_files()
        logger.info(f"共发现 {len(csv_files)} 个 CSV 文件,开始处理,请耐心等待...")
        message_list: List[ChatMessage] = []
        for csv_file in csv_files:
            logger.debug(f"开始处理 CSV 文件: {csv_file}")
            chat_messages = self.load_csv(csv_file)
            message_list.extend(self.group_consecutive_messages(messages=chat_messages))
            logger.debug(f"处理完成: {csv_file}，共加载 {len(chat_messages)} 条消息")
        
        qa_res = self.match_qa(message_list)
        qa_res = [item for item in qa_res if isinstance(item, QaPairV2)]
        
        # 如果启用图片识别，则执行并行处理
        if self.image_processor:
            logger.info("开始执行图片识别流程...")
            qa_res = self._process_images_in_parallel(qa_res)
            logger.info("图片识别流程完成。")

        if self.c.get("clean_dataset", {}).get("enable_clean", False):
            self.clean_strategy.judge(qa_res)  # type: ignore
        
        self.save_result(qa_res)
        self._execute_length_cdf_script()

        logger.success(
            f"聊天记录处理成功，共{len(qa_res)}条，保存到 ./dataset/res_csv/sft/sft-my-img-rec.json"
        )

    def save_result(self, qa_res: List[QaPairV2]):
        """
        重写 save_result 方法以使用新的输出路径，并移除 "images" 键。
        """
        processed_qa_res = []
        for idx, item in enumerate(qa_res):
            item_dict = {
                "id": idx,
                "time": item.time.isoformat() if item.time else None,
                "score": item.score,
                "messages": [
                    {"role": msg.role, "content": msg.content} for msg in item.messages
                ],
                "system": item.system,
            }
            processed_qa_res.append(item_dict)

        output_path = "./dataset/res_csv/sft/sft-my-img-rec.json" 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_qa_res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()