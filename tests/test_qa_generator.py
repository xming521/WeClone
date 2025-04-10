import sys
import os
import pytest
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from make_dataset.models import ChatMessage, CutMessage
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


def test_consecutive_messages_to_csv():
    """
    测试使用DataProcessor的main函数从CSV文件中读取数据，
    应用group_consecutive_messages函数，并将结果保存为CSV
    """
    processor = MockDataProcessor()
    
    # 获取CSV文件列表
    csv_files = processor.get_csv_files()
    
    # 如果没有找到CSV文件，创建一个模拟的CSV文件供测试使用
    if not csv_files:
        print("警告：未找到CSV文件，请确保数据目录中有CSV文件")
        return "无法找到CSV文件"
    
    # 存储所有处理后的消息
    all_grouped_messages = []
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        print(f"处理文件: {csv_file}")
        # 加载CSV文件中的消息
        chat_messages = processor.load_csv(csv_file)
        print(f"加载了 {len(chat_messages)} 条消息")
        
        # 应用group_consecutive_messages函数
        grouped_messages = processor.group_consecutive_messages(messages=chat_messages)
        print(f"分组后得到 {len(grouped_messages)} 条消息")
        
        # 添加到结果列表
        all_grouped_messages.extend(grouped_messages)
    
    # 如果没有处理到任何消息，提前返回
    if not all_grouped_messages:
        print("警告：未处理到任何消息")
        return "未处理到任何消息"
    
    # 将结果转换为DataFrame
    messages_dict = []
    for msg in all_grouped_messages:
        if isinstance(msg, ChatMessage):
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
        elif hasattr(msg, "cut_type"):  # 处理CutMessage对象
            messages_dict.append({
                "id": None,
                "MsgSvrID": None,
                "type_name": msg.cut_type,
                "is_sender": msg.is_sender,
                "talker": None,
                "room_name": None,
                "msg": f"cut",
                "src": None,
                "CreateTime": msg.CreateTime,
            })
    
    # 创建DataFrame
    df = pd.DataFrame(messages_dict)
    
    # 确保输出目录存在
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV文件
    import datetime
    now = datetime.datetime.now()
    output_file = os.path.join(output_dir, f"grouped_messages_.csv")
    # 使用utf-8-sig编码保存，添加BOM标记以解决中文乱码问题
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    # 验证结果
    assert os.path.exists(output_file)
    print(f"已成功保存分组消息到: {output_file}")
    print(f"共保存了 {len(messages_dict)} 条消息")
    
    # 显示前5条消息示例
    if len(messages_dict) > 0:
        print("\n消息示例:")
        for i, msg in enumerate(messages_dict[:5]):
            print(f"{i+1}. {'用户' if msg['is_sender'] == 0 else '对方'}: {msg['msg'][:50]}...")
    
    return output_file


if __name__ == "__main__":
    output_file = test_consecutive_messages_to_csv()
    print(f"测试完成，消息已保存到 {output_file}")
