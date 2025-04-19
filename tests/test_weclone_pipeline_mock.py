import os
import sys
import json
import shutil
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import pandas as pd

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入需要测试的模块
from weclone.data.qa_generator import DataProcessor
from weclone.utils.config import load_config


class TestWeclonePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时目录用于测试
        cls.test_dir = tempfile.mkdtemp()
        cls.test_data_dir = os.path.join(cls.test_dir, "data")
        cls.test_model_dir = os.path.join(cls.test_dir, "model_output")
        cls.test_eval_dir = os.path.join(cls.test_dir, "eval_output")
        
        # 创建必要的目录
        os.makedirs(cls.test_data_dir, exist_ok=True)
        os.makedirs(cls.test_model_dir, exist_ok=True)
        os.makedirs(cls.test_eval_dir, exist_ok=True)
        
        # 创建测试数据集结构
        cls.csv_folder = os.path.join(cls.test_data_dir, "csv")
        os.makedirs(cls.csv_folder, exist_ok=True)
        
        # 创建示例聊天文件夹和CSV文件
        chat_folder = os.path.join(cls.csv_folder, "test_chat")
        os.makedirs(chat_folder, exist_ok=True)
        
        # 创建简单的测试CSV数据
        cls._create_test_csv(os.path.join(chat_folder, "test_chat.csv"))
        
        # 创建测试用的settings.json
        cls._create_test_settings()
        
        # 创建测试用的test_data.json用于模型评估
        cls._create_test_eval_data()
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def _create_test_csv(cls, file_path):
        """创建测试用CSV文件"""
        import pandas as pd
        
        # 创建简单的聊天记录数据
        data = {
            "id": list(range(1, 5)),
            "MsgSvrID": list(range(1001, 1005)),
            "type": ["1", "1", "1", "1"],  # 文本类型
            "is_sender": [0, 1, 0, 1],  # 0=对方发送，1=自己发送
            "talker": ["test_user", "me", "test_user", "me"],
            "room_name": ["", "", "", ""],
            "content": ["你好，请问你是谁？", "我是你的微信助手", "你能帮我做什么？", "我可以回答问题，提供信息和帮助你完成各种任务"],
            "src": ["", "", "", ""],
            "CreateTime": [1609459200, 1609459220, 1609459240, 1609459260]  # 时间戳
        }
        
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    @classmethod
    def _create_test_settings(cls):
        """创建测试用的settings.json"""
        # 简化版的设置文件，只包含测试所需的最小配置
        settings = {
            "train_sft_args": {
                "stage": "sft",
                "dataset": "wechat-sft",
                "dataset_dir": cls.test_data_dir + "/res_csv/sft",
                "lora_target": "query_key_value",
                "lora_rank": 4,
                "lora_dropout": 0.5,
                "overwrite_cache": True,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lr_scheduler_type": "cosine",
                "logging_steps": 1,
                "save_steps": 1,
                "learning_rate": 0.0001,
                "num_train_epochs": 1,
                "plot_loss": False,
                "fp16": False
            },
            "infer_args": {
                "repetition_penalty": 1.2,
                "temperature": 0.5,
                "max_length": 50,
                "top_p": 0.65
            },
            "make_dataset_args": {
                "single_combine_strategy": "time_window",
                "qa_match_strategy": "time_window",
                "single_combine_time_window": 2,
                "qa_match_time_window": 5,
                "prompt_with_history": False
            },
            "common_args": {
                "model_name_or_path": "./chatglm3-6b",  # 假设已有模型
                "adapter_name_or_path": cls.test_model_dir,
                "template": "chatglm3-weclone",
                "finetuning_type": "lora",
                "trust_remote_code": True
            }
        }
        
        # 保存到临时目录
        with open(os.path.join(cls.test_dir, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    
    @classmethod
    def _create_test_eval_data(cls):
        """创建测试用的评估数据"""
        test_data = {
            "questions": [
                ["你好", "你是谁"],
                ["你能做什么"]
            ]
        }
        
        # 确保目录存在
        data_dir = os.path.join(cls.test_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 保存测试数据
        with open(os.path.join(data_dir, "test_data.json"), "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    @patch('weclone.data.qa_generator.DataProcessor.get_csv_files')
    @patch('weclone.data.qa_generator.DataProcessor.load_csv')
    @patch('weclone.data.qa_generator.DataProcessor.save_result')
    def test_qa_generator(self, mock_save_result, mock_load_csv, mock_get_csv_files):
        """测试QA生成器"""
        print("\n测试QA生成器...")
        
        # 准备模拟数据
        from weclone.data.models import ChatMessage
        mock_get_csv_files.return_value = ["test_csv_file.csv"]
        
        # 模拟从CSV加载的消息
        mock_messages = [
            ChatMessage(id=1, MsgSvrID=1001, type_name="文本", is_sender=0, 
                       talker="test_user", room_name="", msg="你好，请问你是谁？", 
                       src="", CreateTime=pd.Timestamp(1609459200, unit='s')),
            ChatMessage(id=2, MsgSvrID=1002, type_name="文本", is_sender=1, 
                       talker="me", room_name="", msg="我是你的微信助手", 
                       src="", CreateTime=pd.Timestamp(1609459220, unit='s')),
            ChatMessage(id=3, MsgSvrID=1003, type_name="文本", is_sender=0, 
                       talker="test_user", room_name="", msg="你能帮我做什么？", 
                       src="", CreateTime=pd.Timestamp(1609459240, unit='s')),
            ChatMessage(id=4, MsgSvrID=1004, type_name="文本", is_sender=1, 
                       talker="me", room_name="", msg="我可以回答问题，提供信息和帮助你完成各种任务", 
                       src="", CreateTime=pd.Timestamp(1609459260, unit='s'))
        ]
        mock_load_csv.return_value = mock_messages
        
        # 创建DataProcessor实例
        with patch('weclone.utils.config.load_config') as mock_load_config:
            # 模拟配置
            mock_config = {
                "single_combine_strategy": "time_window",
                "qa_match_strategy": "time_window",
                "single_combine_time_window": 2,
                "qa_match_time_window": 5,
                "prompt_with_history": False
            }
            mock_load_config.return_value = mock_config
            
            # 执行QA生成
            processor = DataProcessor()
            processor.csv_folder = self.csv_folder  # 设置为测试目录
            processor.main()
            
            # 验证是否调用了预期的方法
            mock_get_csv_files.assert_called_once()
            mock_load_csv.assert_called_once()
            mock_save_result.assert_called_once()
            
            # 验证结果格式
            # 获取保存的结果
            call_args = mock_save_result.call_args[0][0]
            self.assertTrue(isinstance(call_args, list))
            self.assertEqual(len(call_args), 2)  # 应该有两个QA对
            
            # 验证QA对的结构
            for qa in call_args:
                self.assertTrue("instruction" in qa)
                self.assertTrue("output" in qa)
            
            print("QA生成器测试成功")
    
    def test_train_sft(self):
        """测试SFT训练过程"""
        print("\n测试SFT训练过程...")
        # 由于训练需要实际的模型和数据，这里我们只模拟调用
        
        with patch('llamafactory.train.tuner.run_exp') as mock_run_exp:
            # 导入训练模块并运行
            from weclone.train.train_sft import run_exp
            
            
            # 验证是否正确调用了训练函数
            self.assertTrue(mock_run_exp.called)
            print("SFT训练过程测试成功")
    
    def test_api_service(self):
        """测试API服务"""
        print("\n测试API服务...")
        
        # 模拟服务器进程
        with patch('uvicorn.run') as mock_run:
            # 导入API服务模块
            from weclone.server.api_service import main, create_app, ChatModel
            
            # 模拟配置和模型
            with patch('weclone.utils.config.load_config') as mock_load_config:
                mock_config = {"model_path": "test_model_path"}
                mock_load_config.return_value = mock_config
                
                # 模拟ChatModel
                with patch('llamafactory.chat.ChatModel') as MockChatModel:
                    mock_chat_model = MagicMock()
                    MockChatModel.return_value = mock_chat_model
                    
                    # 运行API服务
                    main()
                    
                    # 验证服务是否正确启动
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args[1]
                    self.assertEqual(call_args["host"], "0.0.0.0")
                    self.assertEqual(call_args["port"], 8005)  # 默认端口
                    self.assertEqual(call_args["workers"], 1)
                    
                    print("API服务测试成功")
    
    def test_model_evaluation(self):
        """测试模型评估"""
        print("\n测试模型评估...")
        
        # 模拟OpenAI API调用
        with patch('openai.ChatCompletion.create') as mock_create:
            # 设置模拟返回值
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "这是模型的测试回复"
            mock_create.return_value = mock_response
            
            # 运行评估脚本
            with patch('builtins.open', create=True) as mock_open:
                # 模拟打开测试数据文件
                test_data_content = '{"questions": [["你好", "你是谁"], ["你能做什么"]]}'
                mock_file = MagicMock()
                mock_file.read.return_value = test_data_content
                mock_open.return_value.__enter__.return_value = mock_file
                
                # 导入并运行评估模块
                from weclone.eval.test_model import main
                
                # 执行评估
                main()
                
                # 验证API调用次数（应该是测试问题的数量）
                self.assertEqual(mock_create.call_count, 3)  # 3个测试问题
                
                print("模型评估测试成功")
    
    def test_full_pipeline(self):
        """测试完整流程"""
        print("\n测试完整流程...")
        
        # 这个测试方法会依次调用上面的各个测试方法，模拟完整的流程
        
        # 1. 测试QA生成器
        self.test_qa_generator()
        
        # 2. 测试SFT训练
        self.test_train_sft()
        
        # 3. 测试API服务
        self.test_api_service()
        
        # 4. 测试模型评估
        self.test_model_evaluation()
        
        print("完整流程测试完成")


if __name__ == "__main__":
    unittest.main() 