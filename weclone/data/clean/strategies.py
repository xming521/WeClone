import json
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from langchain_core.prompts import PromptTemplate
from weclone.data.models import QaPair, CutMessage, QaPairScore
from weclone.prompts.clean_data import CLEAN_PROMPT
import os
from weclone.utils.log import logger


@dataclass
class CleaningStrategy(ABC):
    """数据清洗策略的抽象基类"""

    make_dataset_config: Dict

    @abstractmethod
    def clean(self, data: Any) -> Any:
        """
        执行数据清洗操作。

        Args:
            data: 需要清洗的数据。

        Returns:
            清洗后的数据。
        """
        pass


@dataclass
class LLMCleaningStrategy(CleaningStrategy):
    """使用大模型进行数据清洗的策略"""

    def judge(self, data: List[QaPair]) -> None:
        """
        调用llm打分，并将分数直接赋值给传入的QaPair。
        """
        from weclone.core.inference.offline_infer import vllm_infer

        logger.info("开始使用llm对数据打分")
        inputs = []
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            inputs.append(prompt_template.invoke({"id": qa.id, "Q": qa.instruction, "A": qa.output}).text)  # type: ignore
        outputs = vllm_infer(
            inputs,
            self.make_dataset_config["model_name_or_path"],
            template=self.make_dataset_config["template"],
            temperature=0,
            guided_decoding_class=QaPairScore,
            repetition_penalty=1.2,
            bad_words=[r"\n"],
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
                logger.warning(f"Warning: Score not found for QaPair with id {qa.id}. Assigning default score.")

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

    def clean(self) -> str:
        """
        清洗 SFT 数据并返回清洗后的文件路径。
        如果未启用清洗，则返回原始路径。
        """
        config = self.make_dataset_config
        dataset_dir = config["dataset_dir"]
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")

        sft_json_path = os.path.join(dataset_dir, "sft-my.json")
        output_json_path = os.path.join(dataset_dir, "sft-my-l.json")
        accept_score = config.get("clean_dataset", {}).get("llm", {}).get("accept_score", 1)

        if not config.get("clean_dataset", {}).get("enable_clean") or "image" in config.get("include_type", ""):
            logger.info("未启用清洗功能")
            self._update_dataset_info_file(dataset_info_path, new_file_name="sft-my.json")
            return sft_json_path

        try:
            with open(sft_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            filtered_data = [item for item in data if item.get("score", 0) >= accept_score]

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=4)

            logger.success(f"已筛出低于{accept_score}分的数据，共保留 {len(filtered_data)} 条数据")
            self._update_dataset_info_file(dataset_info_path, new_file_name="sft-my-l.json")
            return output_json_path

        except Exception as e:
            logger.error(f"清洗数据失败，使用原始数据: {str(e)}")
            self._update_dataset_info_file(dataset_info_path, new_file_name="sft-my.json")
            return sft_json_path

    def _update_dataset_info_file(self, dataset_info_path: str, new_file_name: str):
        """
        修改 dataset_info.json 文件中的 file_name 字段
        """
        try:
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)

            # 更新所有支持的数据集的 file_name
            for key in ["wechat-sft", "wechat-sft-with-history"]:
                if key in dataset_info:
                    dataset_info[key]["file_name"] = new_file_name

            # 写回文件
            with open(dataset_info_path, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=4, ensure_ascii=False)

            logger.info(f"已更新 dataset_info.json 中的 file_name 为 {new_file_name}")

        except Exception as e:
            logger.warning(f"无法更新 dataset_info.json: {e}")
