import json
from typing import List, Dict
from datetime import datetime

class SFTDatasetBuilder:
    def __init__(self, mbti_type: str = "ENFJ", target_name: str = "鼎伦"):
        self.mbti_type = mbti_type
        self.target_name = target_name
    
    def create_chat_style_system_prompt(self) -> str:
        """创建学习聊天风格的系统提示词"""
        return f"你需要模仿{self.target_name}的日常对话风格和表达习惯。请保持人类身份，不要提及自己是AI。专注于模仿说话风格。"
    
    def create_personality_system_prompt(self) -> str:
        """创建学习性格特征的系统提示词"""
        mbti_traits = {
            "ENFJ": "热情、有同理心、善于激励他人、关心他人成长",
            "INFP": "理想主义、真诚、富有同情心、重视个人价值", 
            "INTJ": "独立、有远见、善于规划、追求卓越",
            "ESFP": "活泼热情、善于社交、注重当下、富有表现力"
        }
        traits = mbti_traits.get(self.mbti_type, mbti_traits["ENFJ"])
        return f"你的MBTI是{self.mbti_type}，性格特征：{traits}。在回答时请自然地体现这些特征。保持人类身份，不要提及自己是AI。"
    
    def convert_mbti_to_wechat_format(self, mbti_data: List[Dict]) -> List[Dict]:
        """将MBTI数据转换为微信数据格式"""
        converted_data = []
        personality_prompt = self.create_personality_system_prompt()
        
        for i, item in enumerate(mbti_data):
            converted_item = {
                "id": len(converted_data),
                "time": datetime.now().isoformat(),
                "score": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": item["instruction"]
                    },
                    {
                        "role": "assistant", 
                        "content": item["output"]
                    }
                ],
                "system": personality_prompt
            }
            converted_data.append(converted_item)
        
        return converted_data
    
    def process_wechat_data(self, wechat_data: List[Dict]) -> List[Dict]:
        """处理微信数据，只更新system字段"""
        processed_data = []
        chat_style_prompt = self.create_chat_style_system_prompt()
        
        for item in wechat_data:
            processed_item = item.copy()
            processed_item["system"] = chat_style_prompt
            processed_data.append(processed_item)
        
        return processed_data
    
    def build_dataset(self, wechat_data: List[Dict], mbti_data: List[Dict]) -> List[Dict]:
        """构建完整数据集"""
        # 处理微信数据
        processed_wechat = self.process_wechat_data(wechat_data)
        
        # 转换MBTI数据
        converted_mbti = self.convert_mbti_to_wechat_format(mbti_data)
        
        # 重新编号
        all_data = processed_wechat + converted_mbti
        for i, item in enumerate(all_data):
            item["id"] = i
        
        print(f"微信数据：{len(processed_wechat)} 条")
        print(f"MBTI数据：{len(converted_mbti)} 条")
        print(f"总计：{len(all_data)} 条")
        
        return all_data
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """保存数据集"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"数据集已保存到：{output_path}")

# 使用示例
def main():
    # 加载原始数据
    with open('dataset/res_csv/sft/sft-my-img-rec.json', 'r', encoding='utf-8') as f:
        wechat_data = json.load(f)
    with open('dataset/mbti/zh_ENFJ_self_awareness.json', 'r', encoding='utf-8') as f:
        mbti_data = json.load(f)

    # 构建数据集
    builder = SFTDatasetBuilder(mbti_type="ENFJ", target_name="Mark")
    dataset = builder.build_dataset(wechat_data, mbti_data)
    
    # 保存数据集
    builder.save_dataset(dataset, "dataset/res_csv/sft/final_sft_dataset.json")

if __name__ == "__main__":
    main()