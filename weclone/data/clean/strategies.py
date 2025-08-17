import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, cast

import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from weclone.core.inference.online_infer import OnlineLLM
from weclone.data.models import QaPair, QaPairScore, QaPairScoreWithId
from weclone.prompts.clean_data import CLEAN_PROMPT, ONLINE_LLM_CLEAN_PROMPT
from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger


@dataclass
class CleaningStrategy(ABC):
    """Abstract base class for data cleaning strategies, but provides common cleaning methods"""

    make_dataset_config: WCMakeDatasetConfig

    @abstractmethod
    def judge(self, data: List[QaPair]) -> None:
        """
        Scoring method, needs to be implemented by subclasses.
        """
        pass

    def clean(self) -> str:
        """
        Filter SFT data based on score and return the final dataset name to use.
        """
        config = self.make_dataset_config
        original_dataset_name = config.dataset
        cleaned_dataset_name = original_dataset_name + "-cleaned"

        dataset_dir = config.dataset_dir
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")

        with open(dataset_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        paths = {
            name: os.path.join(dataset_dir, info.get(name, {}).get("file_name"))
            for name in [original_dataset_name, cleaned_dataset_name]
        }
        original_data_path, cleaned_data_path = paths.values()

        try:
            with open(original_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            accept_score = config.clean_dataset.llm.accept_score
            filtered_data = [item for item in data if item.get("score", 0) >= accept_score]

            if not filtered_data:
                logger.warning("No data retained after cleaning, will use original dataset.")
                return original_dataset_name

            with open(cleaned_data_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            logger.success(
                f"Filtered data below {accept_score} score, retained {len(filtered_data)} items, saved to {cleaned_data_path}"
            )
            return cleaned_dataset_name

        except Exception as e:
            logger.error(f"Error occurred during data cleaning, will use original dataset: {e}")
            return original_dataset_name


@dataclass
class LLMCleaningStrategy(CleaningStrategy):
    """Strategy for data cleaning using large language models"""

    make_dataset_config: WCMakeDatasetConfig

    def judge(self, data: List[QaPair]) -> None:
        """
        Call LLM for scoring and directly assign scores to the input QaPair.
        """
        from weclone.core.inference.offline_infer import vllm_infer

        logger.info("Starting LLM scoring of data")
        inputs = []
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            if qa.images:
                qa.score = 6
            else:
                messages_str = ""
                for msg in qa.messages:
                    if msg.role == "user":
                        messages_str += f"Q: {msg.content}\n"
                    elif msg.role == "assistant":
                        messages_str += f"A: {msg.content}\n"
                prompt_value = prompt_template.invoke({"id": qa.id, "messages": messages_str.strip()})
                inputs.append(prompt_value.to_string())

        parsed_scores, failed_indexs = vllm_infer(
            inputs,
            self.make_dataset_config.model_name_or_path,
            template=self.make_dataset_config.template,
            temperature=0,
            guided_decoding_class=QaPairScore,
            repetition_penalty=1.1,
            enable_thinking=self.make_dataset_config.clean_dataset.llm.enable_thinking,
            cutoff_len=self.make_dataset_config.messages_max_length + 1024,  # add prompt length
            max_new_tokens=1024 if self.make_dataset_config.clean_dataset.llm.enable_thinking else 200,
        )

        # We align scores by iterating only non-image examples and popping from the head of parsed_scores.
        # Build an iterator over parsed results for simplicity and safety.
        parsed_iter = iter(cast(List[QaPairScore | None], parsed_scores))
        non_image_count = 0
        failed_count = 0

        for qa in data:
            if qa.images:
                continue
            non_image_count += 1
            parsed_item = next(parsed_iter, None)
            if parsed_item is None:
                failed_count += 1
                qa.score = 0
            else:
                qa.score = parsed_item.score

        # Sanity check: number of Nones should equal failed_indexs; and total length matches non-image count
        assert failed_count == len(failed_indexs), (
            f"Mismatch: failed_count({failed_count}) != failed_indexs({len(failed_indexs)})"
        )
        assert len(cast(List[QaPairScore | None], parsed_scores)) == non_image_count, (
            f"Mismatch: len(parsed_scores)({len(cast(List[QaPairScore | None], parsed_scores))}) != non_image_count({non_image_count})"
        )

        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)  # Try to fix alignment issues
        distribution_df = pd.DataFrame(  # Merge count and percentage into one DataFrame for printing
            {
                "Count": score_counts,
                "Percentage(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "Score"  # Add column name for the first column: Score
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"LLM scoring distribution:\n{printable_df_str}")


@dataclass
class OlineLLMCleaningStrategy(CleaningStrategy):
    """Strategy for data cleaning using large language models"""

    def judge(self, data: List[QaPair]) -> None:
        config = self.make_dataset_config
        logger.info("Starting online model scoring of data")
        logger.info(f"Using model {config.model_name}")

        client = OnlineLLM(
            api_key=config.llm_api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            default_system=config.default_system,
        )
        prompt_template = PromptTemplate.from_template(ONLINE_LLM_CLEAN_PROMPT)

        parsed_scores = []
        clean_batch_size = config.clean_batch_size

        for i in tqdm(range(0, len(data), clean_batch_size), desc="Online model scoring progress"):
            batch = data[i : i + clean_batch_size]
            # Construct qa_list for current batch
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
            # Fill template
            prompt_text = prompt_template.invoke({"qa_list": qa_list_json}).text
            try:
                response = client.chat(prompt_text, temperature=0)
                result_text = response.choices[0].message.content
                # print("Model response:",result_text)
                # If there is <think> … </think>, keep only the content after </think>
                if "</think>" in result_text:
                    result_text = result_text.split("</think>", 1)[1]
                # Remove leading and trailing ```json or ``` code block markers
                result_text = re.sub(r"^```json\s*|```$", "", result_text.strip(), flags=re.MULTILINE)
                # Skip if occasional parsing failures occur
                try:
                    score_list = json.loads(result_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed, skipping this batch: {e}\nContent: {result_text}")
                    continue

                for item in score_list:
                    parsed_scores.append(QaPairScoreWithId(**item))
            except Exception as e:
                ids_in_batch = [qa["id"] for qa in qa_list]
                logger.error(
                    f"Failed to call online model or parse result, current batch QA ID list: {ids_in_batch}, error: {str(e)}"
                )

        score_map = {score.id: score.score for score in parsed_scores}
        for qa in data:
            if qa.id in score_map:
                qa.score = score_map[qa.id]
            else:
                logger.warning(f"No score obtained for QA ID {qa.id}, default assigned 0")
                qa.score = 0

        # Calculate score distribution and print logs (consistent with local version)
        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)
        distribution_df = pd.DataFrame(
            {
                "Count": score_counts,
                "Percentage(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "Score"
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"Online model scoring distribution:\n{printable_df_str}")
