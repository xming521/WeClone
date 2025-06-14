import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from weclone.core.inference.online_infer import OnlineLLM
from weclone.data.models import QaPair, QaPairScore
from weclone.prompts.clean_data import CLEAN_PROMPT, ONLINE_LLM_CLEAN_PROMPT
from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger


@dataclass
class CleaningStrategy(ABC):
    """数据清洗策略的抽象基类，但提供通用的清洗方法"""

    make_dataset_config: WCMakeDatasetConfig

    @abstractmethod
    def judge(self, data: List[QaPair]) -> None:
        """
        打分方法是抽象的，强制每个子类根据自己的方式去实现。
        """
        pass

    def clean(self) -> str:
        """
        通用策略
        根据score筛选SFT数据，并返回最终应使用的dataset名称。
        """
        config = self.make_dataset_config
        original_dataset_name = config.dataset
        cleaned_dataset_name = "chat-sft-cleaned"

        if not config.clean_dataset.enable_clean or "image" in config.include_type:
            logger.info("数据清洗未启用或包含图像，将使用原始数据集。")
            return original_dataset_name

        dataset_dir = config.dataset_dir
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")

        # 获取文件名称
        try:
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            paths = {
                name: os.path.join(dataset_dir, info.get(name, {}).get("file_name"))
                for name in [original_dataset_name, cleaned_dataset_name]
            }
            original_data_path, cleaned_data_path = paths.values()
            if not all(paths.values()):
                raise ValueError(f"缺失 '{original_dataset_name}' 或 '{cleaned_dataset_name}' 文件配置。")
        except Exception as e:
            logger.error(f"加载 dataset_info.json 出错: {e}，将使用原始数据集。")
            return original_dataset_name

        # 执行清洗流程
        logger.info(f"数据清洗已启用，将从 '{original_data_path}' 读取数据...")
        try:
            if not os.path.exists(original_data_path):
                logger.error(f"原始数据文件 '{original_data_path}' 不存在，清洗中止。")
                return original_dataset_name
            with open(original_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            accept_score = config.clean_dataset.llm.accept_score
            filtered_data = [item for item in data if item.get("score", 0) >= accept_score]

            if not filtered_data:
                logger.warning("清洗后无数据保留，将使用原始数据集。")
                return original_dataset_name

            with open(cleaned_data_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            logger.success(
                f"已筛出低于 {accept_score} 分的数据，保留 {len(filtered_data)} 条，保存至 {cleaned_data_path}"
            )
            return cleaned_dataset_name

        except Exception as e:
            logger.error(f"数据清洗过程中发生错误，将使用原始数据集: {e}")
            return original_dataset_name


@dataclass
class LLMCleaningStrategy(CleaningStrategy):
    """使用大模型进行数据清洗的策略"""

    make_dataset_config: WCMakeDatasetConfig

    def judge(self, data: List[QaPair]) -> None:
        """
        调用llm打分，并将分数直接赋值给传入的QaPair。
        """
        from weclone.core.inference.offline_infer import vllm_infer

        logger.info("开始使用llm对数据打分")
        inputs = []
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            messages_str = ""
            for msg in qa.messages:
                if msg.role == "user":
                    messages_str += f"Q: {msg.content}\n"
                elif msg.role == "assistant":
                    messages_str += f"A: {msg.content}\n"
            prompt_value = prompt_template.invoke({"id": qa.id, "messages": messages_str.strip()})
            inputs.append(prompt_value.to_string())

        outputs = vllm_infer(
            inputs,
            self.make_dataset_config.model_name_or_path,
            template=self.make_dataset_config.template,
            temperature=0,
            guided_decoding_class=QaPairScore,
            repetition_penalty=1.5,
            bad_words=[r"\n"],
            enable_thinking=False,
            cutoff_len=self.make_dataset_config.messages_max_length + 1024,  # add prompt length
            max_new_tokens=100,
        )

        parsed_scores: List[QaPairScore] = []
        for result in outputs:
            try:
                score_data = json.loads(result.outputs[0].text)
                qa_score = QaPairScore(**score_data)
                parsed_scores.append(qa_score)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON: {result.outputs[0].text}")

        score_map = {score.id: score.score for score in parsed_scores}
        for qa in data:
            if qa.id in score_map:
                qa.score = score_map[qa.id]
            else:
                logger.warning(
                    f"Warning: Score not found for QaPair with id {qa.id}. Assigning default score."
                )

        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)  # 尝试修正对齐问题
        distribution_df = pd.DataFrame(  # 合并数量和百分比到一个 DataFrame 中以便打印
            {
                "数量": score_counts,
                "占比(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "分数"  # 给第一列加上列名：分数
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"llm打分分数分布情况:\n{printable_df_str}")


@dataclass
class OlineLLMCleaningStrategy(CleaningStrategy):
    """使用大模型进行数据清洗的策略"""

    def judge(self, data: List[QaPair]) -> None:
        config = self.make_dataset_config
        logger.info("开始使用在线模型对数据打分")
        logger.info(f"使用模型 {config.model_name}")

        client = OnlineLLM(
            api_key=config.llm_api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            default_system=config.default_system,
        )
        prompt_template = PromptTemplate.from_template(ONLINE_LLM_CLEAN_PROMPT)

        parsed_scores = []
        clean_batch_size = config.clean_batch_size

        for i in tqdm(range(0, len(data), clean_batch_size), desc="在线模型评分进度"):
            batch = data[i : i + clean_batch_size]
            # 构造当前批次的 qa_list
            # qa_list = [{"id": qa.id, "Q": qa.instruction, "A": qa.output} for qa in batch]
            qa_list = [
                {
                    "id": qa.id,
                    "Q": next((msg.content for msg in qa.messages if msg.role == "user"), ""),
                    "A": next((msg.content for msg in qa.messages if msg.role == "assistant"), ""),
                }
                for qa in batch
            ]
            qa_list_json = json.dumps(qa_list, ensure_ascii=False)
            # 填充模板
            prompt_text = prompt_template.invoke({"qa_list": qa_list_json}).text
            try:
                response = client.chat(prompt_text)
                result_text = response.choices[0].message.content
                # print("大模型返回：",result_text)
                # 如果有 <think> … </think>，只保留 </think> 之后的内容
                if "</think>" in result_text:
                    result_text = result_text.split("</think>", 1)[1]
                # 去掉开头和结尾的 ```json 或 ``` 等代码块标记
                result_text = re.sub(r"^```json\s*|```$", "", result_text.strip(), flags=re.MULTILINE)
                # 如果偶尔的几次解析失败就跳过
                try:
                    score_list = json.loads(result_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析失败，跳过本批次: {e}\n内容：{result_text}")
                    continue

                for item in score_list:
                    parsed_scores.append(QaPairScore(**item))
            except Exception as e:
                ids_in_batch = [qa["id"] for qa in qa_list]
                logger.error(
                    f"调用在线模型或解析结果失败，当前 batch QA ID 列表: {ids_in_batch}，错误信息: {str(e)}"
                )

        score_map = {score.id: score.score for score in parsed_scores}
        for qa in data:
            if qa.id in score_map:
                qa.score = score_map[qa.id]
            else:
                logger.warning(f"未获取到QA ID {qa.id}的分数，默认赋值0")
                qa.score = 0

        # 统计分数分布，打印日志（和本地版本保持一致）
        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)
        distribution_df = pd.DataFrame(
            {
                "数量": score_counts,
                "占比(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "分数"
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"在线模型打分分数分布情况:\n{printable_df_str}")
