import json
import random
from datetime import datetime
from typing import Dict, List


class SFTDatasetBuilder:
    """
    一个用于构建SFT（监督微调）数据集的工具类。
    它能将个人聊天记录（用于学习风格）和MBTI问答对（用于学习性格）
    合并成一个统一的、格式化的、随机打乱的数据集。
    """

    def __init__(self, mbti_type: str = "ENFJ", target_name: str = "Mark"):
        """
        初始化构建器。

        Args:
            mbti_type (str): 目标人物的MBTI类型。
            target_name (str): 目标人物的名字。
        """
        self.mbti_type = mbti_type
        self.target_name = target_name
        # 将MBTI特质描述作为类属性，方便管理和扩展
        self.mbti_traits_db = {
            "ENFJ": "你热情、有同理心、善于激励他人、能敏锐地察觉他人的情绪，并总是乐于给予支持和鼓励。",
            "INFP": "你是一个理想主义者，真诚、富有同情心，非常重视个人价值和内心世界。",
            "INTJ": "你思想独立、有远见、善于规划和分析，总是追求知识和能力的卓越。",
            "ESFP": "你活泼热情、善于社交、享受当下，富有表现力，是大家眼中的开心果。",
        }
        print(f"角色设定：名字={self.target_name}, MBTI={self.mbti_type}")

    def create_unified_system_prompt(self) -> str:
        """
        创建统一的、融合了性格与风格的“角色卡”式系统提示词。
        """
        traits = self.mbti_traits_db.get(self.mbti_type, self.mbti_traits_db["ENFJ"])

        unified_prompt = f"""你现在是{self.target_name}，你需要完全沉浸在这个角色中进行对话。

# 核心角色设定
- **身份**: 你是{self.target_name}，一个{self.mbti_type}类型的人。
- **性格**: {traits} 在对话中，请自然地体现这些特征。

# 沟通风格 (重要)
- **模仿重点**: 你的主要任务是模仿{self.target_name}的日常对话风格、语气和表达习惯。
- **对话方式**: 请使用口语化的、像与朋友聊天一样的语气，而不是书面化的AI语言。

# 行为准则
- 绝对不能承认或暗示自己是一个AI或模型。你的记忆和知识都来自于你作为{self.target_name}的个人经历。
- 当被问及不知道的事情时，像真人一样回答“我不太清楚诶”或“让我想想”，而不是直接拒绝。"""

        return unified_prompt.strip()

    def build_dataset(self, wechat_data: List[Dict], mbti_data: List[Dict]) -> List[Dict]:
        """
        构建、合并并随机化整个数据集。

        Args:
            wechat_data (List[Dict]): 格式化的微信聊天记录数据。
            mbti_data (List[Dict]): 包含 "instruction" 和 "output" 的MBTI问答对。

        Returns:
            List[Dict]: 最终的、可用于SFT的数据集。
        """
        # 1. 创建唯一的系统提示词
        unified_prompt = self.create_unified_system_prompt()
        print("已生成统一的System Prompt，将应用于所有数据。")

        all_data = []

        # 获取一个真实的时间戳，用于填充MBTI数据，使其时间上看起来更连贯
        last_real_timestamp = datetime.now().isoformat()
        if wechat_data:
            # 尝试从最后一条微信记录获取时间戳
            last_real_timestamp = wechat_data[-1].get("time", last_real_timestamp)

        # 2. 处理微信数据，应用统一的Prompt
        for item in wechat_data:
            processed_item = item.copy()
            processed_item["system"] = unified_prompt
            all_data.append(processed_item)

        # 3. 处理并转换MBTI数据，应用统一的Prompt
        for item in mbti_data:
            converted_item = {
                "id": -1,
                "time": last_real_timestamp,
                "score": 0,
                "messages": [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]},
                ],
                "system": unified_prompt,
            }
            all_data.append(converted_item)

        # 4. 随机打乱所有数据（关键步骤，确保训练稳定）
        print(f"\n合并了 {len(wechat_data)} 条微信数据和 {len(mbti_data)} 条MBTI数据。")
        print(f"总计 {len(all_data)} 条数据，正在进行随机打乱...")
        random.shuffle(all_data)

        # 5. 在打乱后，统一重新编号所有数据的ID，确保ID是连续的
        for i, item in enumerate(all_data):
            item["id"] = i

        print("数据集已混合并重新编号完毕。")
        return all_data

    def save_dataset(self, dataset: List[Dict], output_path: str):
        """
        将最终的数据集保存到JSON文件。

        Args:
            dataset (List[Dict]): 待保存的数据集。
            output_path (str): 输出文件路径。
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # 使用indent=2以减小文件体积，同时保持基本可读性
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"\n数据集已成功保存到：{output_path}")
        except IOError as e:
            print(f"错误：无法写入文件 {output_path}。错误信息: {e}")


def main():
    """
    主执行函数，负责整个流程的调用。
    """
    # --- 配置区 ---
    # 在这里修改你的输入输出文件路径和角色信息
    WECHAT_INPUT_PATH = r"D:\Desktop\Just_for_fun\a\WeClone\dataset/res_csv/sft/sft-my-img-rec.json"
    MBTI_INPUT_PATH = r"D:\Desktop\Just_for_fun\a\WeClone\dataset/mbti/zh_ENFJ_self_awareness.json"
    FINAL_OUTPUT_PATH = r"D:\Desktop\Just_for_fun\a\WeClone\dataset/res_csv/sft/sft-my-mbti.json"
    TARGET_NAME = "Mark"
    MBTI_TYPE = "ENFJ"
    # --- 配置区结束 ---

    print("--- 开始构建SFT数据集 ---")

    # 加载原始数据
    try:
        print(f"正在从 {WECHAT_INPUT_PATH} 加载微信数据...")
        with open(WECHAT_INPUT_PATH, "r", encoding="utf-8") as f:
            wechat_data = json.load(f)

        print(f"正在从 {MBTI_INPUT_PATH} 加载MBTI数据...")
        with open(MBTI_INPUT_PATH, "r", encoding="utf-8") as f:
            mbti_data = json.load(f)
    except FileNotFoundError as e:
        print(f"\n错误：找不到输入文件 {e.filename}。请检查路径是否正确。")
        return
    except json.JSONDecodeError as e:
        print(f"\n错误：解析JSON文件时出错。请检查文件格式是否正确。错误信息: {e}")
        return

    # 初始化构建器
    builder = SFTDatasetBuilder(mbti_type=MBTI_TYPE, target_name=TARGET_NAME)

    # 执行数据集构建
    final_dataset = builder.build_dataset(wechat_data, mbti_data)

    # 保存最终的数据集
    builder.save_dataset(final_dataset, FINAL_OUTPUT_PATH)

    print("\n--- 数据集构建完成 ---")


if __name__ == "__main__":
    main()
