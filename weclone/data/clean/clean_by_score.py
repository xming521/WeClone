import os
import json
from weclone.utils.config import load_config
from weclone.utils.log import logger


def clean_sft_data() -> str:
    """
    清洗 SFT 数据并返回清洗后的文件路径。
    如果未启用清洗，则返回原始路径。
    """
    config = load_config(arg_type="make_dataset")
    sft_json_path = os.path.join(config["dataset_dir"], "sft-my.json")
    output_json_path = os.path.join(config["dataset_dir"], "sft-my-l.json")
    accept_score = config.get("clean_dataset", {}).get("llm", {}).get("accept_score", 1)

    if not config.get("clean_dataset", {}).get("enable_clean"):
        logger.info("未启用清洗功能")
        return sft_json_path

    try:
        with open(sft_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        filtered_data = [item for item in data if item.get("score", 1) >= accept_score]

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        logger.success(f"已筛出低于{accept_score}分的数据，共保留 {len(filtered_data)} 条数据")
        return output_json_path

    except Exception as e:
        logger.error(f"清洗数据失败，使用原始数据: {str(e)}")
        return sft_json_path


if __name__ == "__main__":
    clean_sft_data()
