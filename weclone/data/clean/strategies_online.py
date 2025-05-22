import re
import json
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
from langchain_core.prompts import PromptTemplate
from weclone.data.models import QaPair, QaPairScore
from weclone.prompts.clean_data import CLEAN_PROMPT,ONLINE_LLM_CLEAN_PROMPT
from weclone.core.inference.online_infer import OnlineLLM
from weclone.utils.log import logger

@dataclass
class CleaningStrategy(ABC):
    """数据清洗策略的抽象基类"""

    make_dataset_config: Dict

    @abstractmethod
    def clean(self, data: Any) -> Any:
        pass

@dataclass
class OlineLLMCleaningStrategy(CleaningStrategy):
    """使用大模型进行数据清洗的策略"""

    def judge(self, data: List[QaPair]) -> None:
        logger.info("开始使用在线模型对数据打分")

        logger.info(f"使用模型 {self.make_dataset_config.get('model_name', '')}")

        client = OnlineLLM(
            api_key = self.make_dataset_config.get("llm_api_key"),
            base_url = self.make_dataset_config.get("base_url"),
            model_name = self.make_dataset_config.get("model_name"),
            default_system = self.make_dataset_config.get("default_system")
        )
        prompt_template = PromptTemplate.from_template(ONLINE_LLM_CLEAN_PROMPT)

        parsed_scores = []
        clean_batch_size = int(self.make_dataset_config.get("clean_batch_size", 10)) 
        for i in tqdm(range(0, len(data), clean_batch_size), desc="在线模型评分进度"):
            batch = data[i : i + clean_batch_size]
            # 构造当前批次的 qa_list
            qa_list = [
                {"id": qa.id, "Q": qa.instruction, "A": qa.output}
                for qa in batch
            ]
            qa_list_json = json.dumps(qa_list, ensure_ascii=False)
            # 填充模板
            prompt_text = prompt_template.invoke({
                "qa_list": qa_list_json
            }).text
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
                logger.error(f"调用在线模型或解析结果失败，QA ID {qa.id}: {str(e)}")

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
        distribution_df = pd.DataFrame({
            "数量": score_counts,
            "占比(%)": score_percentages.round(2),
        })
        distribution_df.index.name = "分数"
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"在线模型打分分数分布情况:\n{printable_df_str}")

    def clean(self, data: List[QaPair]) -> List[QaPair]:
        """
        根据打分结果，删除分数低于阈值的数据。
        """
        threshold = self.make_dataset_config.get("clean_dataset", {}).get("llm", {}).get("accept_score", 1)
        return [
            qa
            for qa in data
            if qa.score is not None and qa.score >= threshold
        ]
