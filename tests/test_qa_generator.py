import sys
import os
import pytest
from datetime import datetime, timedelta

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from make_dataset.models import ChatMessage
from make_dataset.qa_generator import DataProcessor

# 将当前工作目录更改为项目根目录
os.chdir(root_dir)

# # 测试数据处理器类的初始化和配置加载
# def test_data_processor_init():
#     """测试DataProcessor初始化"""
#     processor = DataProcessor()
#     assert processor.csv_folder == "./data/csv"
#     assert "文本" not in processor.skip_type_list
#     assert len(processor.type_list) == 8


class MockDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()


@pytest.fixture
def processor():
    """创建一个测试用的处理器实例"""
    return MockDataProcessor()


def test_empty_messages(processor):
    """测试空消息列表的情况"""
    messages = []
    result = processor.group_consecutive_messages(messages)
    assert result == []


def test_single_message(processor):
    """测试单条消息的情况"""
    now = datetime.now()
    message = ChatMessage(
        id=1,
        MsgSvrID=1001,
        type_name="文本",
        is_sender=0,
        talker="user1",
        room_name="testroom",
        msg="你好",
        src="",
        CreateTime=now,
    )

    result = processor.group_consecutive_messages([message])
    assert len(result) == 1
    assert result[0].msg == "你好"


def test_consecutive_messages_same_sender(processor):
    """测试同一发送者的连续消息"""
    now = datetime.now()
    messages = [
        ChatMessage(
            id=1,
            MsgSvrID=1001,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="你好",
            src="",
            CreateTime=now,
        ),
        ChatMessage(
            id=2,
            MsgSvrID=1002,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="最近怎么样",
            src="",
            CreateTime=now + timedelta(minutes=5),
        ),
        ChatMessage(
            id=3,
            MsgSvrID=1003,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="我想问个问题",
            src="",
            CreateTime=now + timedelta(minutes=10),
        ),
    ]

    result = processor.group_consecutive_messages(messages)
    assert len(result) == 1
    assert result[0].msg == "你好，最近怎么样，我想问个问题"


def test_messages_different_senders(processor):
    """测试不同发送者的消息"""
    now = datetime.now()
    messages = [
        ChatMessage(
            id=1,
            MsgSvrID=1001,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="你好",
            src="",
            CreateTime=now,
        ),
        ChatMessage(
            id=2,
            MsgSvrID=1002,
            type_name="文本",
            is_sender=1,
            talker="user2",
            room_name="testroom",
            msg="你好，有什么可以帮你的",
            src="",
            CreateTime=now + timedelta(minutes=5),
        ),
        ChatMessage(
            id=3,
            MsgSvrID=1003,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="我想问个问题",
            src="",
            CreateTime=now + timedelta(minutes=10),
        ),
    ]

    result = processor.group_consecutive_messages(messages)
    assert len(result) == 3
    assert result[0].msg == "你好"
    assert result[1].msg == "你好，有什么可以帮你的"
    assert result[2].msg == "我想问个问题"


def test_skip_non_text_messages(processor):
    """测试跳过非文本消息"""
    now = datetime.now()
    messages = [
        ChatMessage(
            id=1,
            MsgSvrID=1001,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="你好",
            src="",
            CreateTime=now,
        ),
        ChatMessage(
            id=2,
            MsgSvrID=1002,
            type_name="图片",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="",
            src="image.jpg",
            CreateTime=now + timedelta(minutes=1),
        ),
        ChatMessage(
            id=3,
            MsgSvrID=1003,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="看到图片了吗",
            src="",
            CreateTime=now + timedelta(minutes=2),
        ),
    ]

    result = processor.group_consecutive_messages(messages)
    assert len(result) == 1
    assert result[0].msg == "你好，看到图片了吗"


def test_time_window_limit(processor):
    """测试时间窗口限制（超过1小时的消息不会合并）"""
    now = datetime.now()
    messages = [
        ChatMessage(
            id=1,
            MsgSvrID=1001,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="你好",
            src="",
            CreateTime=now,
        ),
        ChatMessage(
            id=2,
            MsgSvrID=1002,
            type_name="文本",
            is_sender=0,
            talker="user1",
            room_name="testroom",
            msg="晚上好",
            src="",
            CreateTime=now + timedelta(hours=2),  # 超过1小时
        ),
    ]

    result = processor.group_consecutive_messages(messages)
    assert len(result) == 2
    assert result[0].msg == "你好"
    assert result[1].msg == "晚上好"


if __name__ == "__main__":
    import pandas as pd
    import os
    from datetime import datetime
    
    # 创建一个测试函数，将group_consecutive_messages的结果转为DataFrame并保存为CSV
    def test_save_grouped_messages_to_csv(processor):
        """测试将group_consecutive_messages的结果转换为DataFrame并保存为CSV"""
        now = datetime.now()
        messages = [
            ChatMessage(
                id=1,
                MsgSvrID=1001,
                type_name="文本",
                is_sender=0,
                talker="user1",
                room_name="testroom",
                msg="你好",
                src="",
                CreateTime=now,
            ),
            ChatMessage(
                id=2,
                MsgSvrID=1002,
                type_name="文本",
                is_sender=0,
                talker="user1",
                room_name="testroom",
                msg="这是测试消息",
                src="",
                CreateTime=now + timedelta(minutes=10),
            ),
            ChatMessage(
                id=3,
                MsgSvrID=1003,
                type_name="文本",
                is_sender=1,
                talker="user2",
                room_name="testroom",
                msg="收到了",
                src="",
                CreateTime=now + timedelta(minutes=20),
            ),
        ]
        
        # 获取分组后的消息
        grouped_messages = processor.group_consecutive_messages(messages)
        
        # 将ChatMessage对象转换为字典列表
        messages_dict = []
        for msg in grouped_messages:
            messages_dict.append({
                "id": msg.id,
                "MsgSvrID": msg.MsgSvrID,
                "type_name": msg.type_name,
                "is_sender": msg.is_sender,
                "talker": msg.talker,
                "room_name": msg.room_name,
                "msg": msg.msg,
                "src": msg.src,
                "CreateTime": msg.CreateTime,
            })
        
        # 创建DataFrame
        df = pd.DataFrame(messages_dict)
        
        # 确保输出目录存在
        output_dir = "./test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV文件
        output_file = os.path.join(output_dir, f"grouped_messages_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(output_file, index=False, encoding="utf-8")
        
        # 验证文件已创建
        assert os.path.exists(output_file)
        
        # 读取CSV文件并验证内容
        loaded_df = pd.read_csv(output_file)
        assert len(loaded_df) == len(grouped_messages)
        assert loaded_df.iloc[0]["msg"] == "你好，这是测试消息"  # 验证第一条消息已合并
        
        print(f"已成功保存分组消息到: {output_file}")
        return output_file
