import csv
import json
import os

import pandas as pd
from collections import deque

csv_folder = './data/csv'
# csv_folder = './data/test'
print(f'当前处理目录{csv_folder}')


def handle_pt_csv(csvfile):
    chat_df = pd.read_csv(csvfile)
    # 选择type_name为文本的行、is_sender为1的行
    chat_df = chat_df[chat_df['type_name'] == '文本']
    chat_df = chat_df[chat_df['is_sender'] == 1]
    # 对每一行的content进行处理 转为dict 再取'msg'字段
    chat_df['content'] = chat_df['content'].apply(lambda x: json.loads(x)['msg'])
    # 如果content 包含 手机号、身份证号、邮箱、网址则删除这行
    chat_df = chat_df[~chat_df['content'].str.contains('1\d{10}')]
    chat_df = chat_df[~chat_df['content'].str.contains('\d{18}')]
    chat_df = chat_df[~chat_df['content'].str.contains('\w+@\w+')]
    chat_df = chat_df[~chat_df['content'].str.contains('http')]
    chat_df = chat_df[~chat_df['content'].str.contains(r'\\xa0')]
    chat_df = chat_df[~chat_df['content'].str.contains(r'\\u')]

    # 纯content
    chat_df = chat_df['content']
    chat_df = chat_df.dropna()

    return chat_df


def make_pt_dataset():
    csv_res = []
    # csv文件夹里全是不同聊天对象文件夹 每个文件夹里是csv文件 先遍历不同聊天对象文件夹 再遍历聊天对象的csv文件
    for chat_obj_folder in os.listdir(csv_folder):
        chat_obj_folder_path = os.path.join(csv_folder, chat_obj_folder)
        for csvfile in os.listdir(chat_obj_folder_path):
            csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
            chat_df = handle_pt_csv(csvfile_path)
            csv_res.append(chat_df)

    csv_res = pd.concat(csv_res)
    csv_res = csv_res.apply(lambda x: {'c': x})  # 设置数据集prompt键为c

    csv_res.to_json('./data/res_csv/pt-my.json', orient='records', force_ascii=False)


def handle_sft_csv(csvfile):
    chat_df = pd.read_csv(csvfile)
    blocked_words = json.load(open('./make_dataset/blocked_words.json', encoding='utf-8'))['blocked_words']
    # 选择type_name为文本的行、is_sender为1的行
    # 需要保留的type_name字段名
    type_list = ['文本', '图片', '卡片式链接', '合并转发的聊天记录', '视频', '语言', '未知', '分享的小程序']
    chat_df = chat_df[chat_df['type_name'].isin(values=type_list)]

    # 对每一行的content进行处理 转为dict 再取'msg'字段
    chat_df['content'] = chat_df['content'].apply(func=lambda x: json.loads(x)['msg'])
    # 如果type_name为文本 并且content 包含 手机号、身份证号、邮箱、网址则删除这行 用循环删除
    for i in chat_df.index:
        if chat_df.loc[i, 'type_name'] == '文本':
            if ('1\d{10}' in chat_df.loc[i, 'content'] or
                '\d{18}' in chat_df.loc[i, 'content'] or
                '\w+@\w+' in chat_df.loc[i, 'content'] or
                'http' in chat_df.loc[i, 'content'] or
                r'\\xa0' in chat_df.loc[i, 'content'] or
                    r'\\u' in chat_df.loc[i, 'content']):
                chat_df = chat_df.drop(index=i)
                continue
            for blocked_word in blocked_words:
                if blocked_word in chat_df.loc[i, 'content']:
                    chat_df = chat_df.drop(index=i)
                    break
        else:
            # content赋值为空
            chat_df.loc[i, 'content'] = ''

    chat_df = chat_df[['is_sender', 'type_name', 'content', 'CreateTime']]
    chat_df = chat_df.dropna()

    # 时间格式 2021-07-07 10:27:23
    # 遍历行 相同is_sender的行合并content（）遇到不同is_sender就重新开始
    # CreateTime字段保留最后的CreateTime
    chat_df['CreateTime'] = pd.to_datetime(chat_df['CreateTime'])
    type_list.remove('文本')
    skip_list = type_list
    res_df = []
    last_is_sender = chat_df.iloc[0]['is_sender']
    last_content: str = chat_df.iloc[0]['content']
    last_CreateTime = chat_df.iloc[0]['CreateTime']
    # 超时处理 半天没说话就重新开始
    # 注意这里只是处理了组装成一个句子 最后封装对话、配对在make_sft_dataset

    # 遇到图片 连接 直接封装成一个句子
    for i, row in chat_df.iterrows():
        if row['type_name'] in skip_list:
            if last_content != '':
                if last_content[-1] == '，':
                    last_content = last_content[:-1] + '。'
                elif last_content[-1] not in ['。', '！', '？', '…', '.']:
                    last_content += '。'
                res_df.append({'is_sender': last_is_sender, 'content': last_content, 'CreateTime': last_CreateTime})
                last_CreateTime = row['CreateTime']
                last_content = ''
            # cut表示被skip字段截断
            res_df.append({'is_sender': row['is_sender'], 'content': 'cut', 'CreateTime': row['CreateTime']})
            continue
        if last_content == '':  # 重新开始
            last_content = row['content']
            last_is_sender = row['is_sender']
            last_CreateTime = row['CreateTime']
            continue
        if row['is_sender'] == last_is_sender:
            if row['CreateTime'] - last_CreateTime > pd.Timedelta(value='1h'):
                # 如果超时 前面的添加到res_df 并重新开始
                if last_content[-1] == '，':
                    last_content = last_content[:-1] + '。'
                elif last_content[-1] not in ['。', '！', '？', '…', '.']:
                    last_content += '。'
                res_df.append({'is_sender': last_is_sender, 'content': last_content, 'CreateTime': last_CreateTime})
                last_content = row['content']
                last_CreateTime = row['CreateTime']
                continue
            # 如果content的结尾没有标点符号则添加逗号，最后结尾是句号
            if last_content[-1] not in ['。', '！', '？', '…', '，']:
                last_content += '，'
            last_content = last_content + row['content']
            last_CreateTime = row['CreateTime']
        else:
            if last_content[-1] == '，':
                last_content = last_content[:-1] + '。'
            elif last_content[-1] not in ['。', '！', '？', '…', '.']:
                last_content += '。'
            res_df.append({'is_sender': last_is_sender, 'content': last_content, 'CreateTime': last_CreateTime})
            last_is_sender = row['is_sender']
            last_content = row['content']
            last_CreateTime = row['CreateTime']
    res_df = pd.DataFrame(res_df)
    return res_df


def make_sft_dataset():

    #     [
    #   {
    #     "instruction": "用户指令（必填）",
    #     "input": "用户输入（选填）",
    #     "output": "模型回答（必填）",
    #     "system": "系统提示词（选填）",
    #     "history": [
    #       ["第一轮指令（选填）", "第一轮回答（选填）"],
    #       ["第二轮指令（选填）", "第二轮回答（选填）"]
    #     ]
    #   }
    # ]

    csv_concat = []
    csv_res = []
    # csv文件夹里全是不同聊天对象文件夹 每个文件夹里是csv文件 先遍历不同聊天对象文件夹 再遍历聊天对象的csv文件
    for chat_obj_folder in os.listdir(csv_folder):
        chat_obj_folder_path = os.path.join(csv_folder, chat_obj_folder)
        for csvfile in os.listdir(chat_obj_folder_path):
            csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
            chat_df = handle_sft_csv(csvfile_path)
            csv_concat.append(chat_df)

    csv_concat = pd.concat(csv_concat)
    # csv_res里is_sender必须是01 01 01 的顺序 csv_concat里不一定是01 01
    # 相差超过1小时的时间戳分为不同的对话
    # temp_res为一个长度为2的队列
    temp_res = deque(maxlen=2)
    # 6种情况
    # temp_res 为空  遇到 0入队 遇到1不处理 遇到cut不处理
    # temp_res 有0  遇到0清空队列再入队 遇到1相差超过1小时清空队列 没有相差一小时入队再全部出队 遇到cut清空队列

    # 选最长的做为问题的结果？

    for i, row in csv_concat.iterrows():
        if len(temp_res) == 0:
            if row['content'] == 'cut':
                continue
            if row['is_sender'] == 0:
                temp_res.append(row['content'])
                last_CreateTime = row['CreateTime']
            else:
                continue
        elif len(temp_res) == 1:
            if row['content'] == 'cut':
                temp_res.clear()
                last_CreateTime = row['CreateTime']
            elif row['is_sender'] == 0:
                # 遇到0 清空队列再入队
                temp_res.clear()
                temp_res.append(row['content'])
                last_CreateTime = row['CreateTime']
            else:
                if row['CreateTime'] - last_CreateTime > pd.Timedelta('1h'):
                    # 相差超过1小时清空队列
                    temp_res.clear()
                    last_CreateTime = row['CreateTime']
                else:
                    # 没有相差一小时入队再全部出队
                    temp_res.append(row['content'])
                    temp_output_list = temp_res[1].split('，')
                    output = max(temp_output_list, key=len)# 只选选最长的回答作为最终数据
                    if output[-1] == '。':
                        output = output[:-1]
                    csv_res.append({'instruction': temp_res[0], 'output': output})
                    temp_res.clear()
                    last_CreateTime = row['CreateTime']

    csv_res_df = pd.DataFrame(csv_res)
    print(f'数据量：{csv_res_df.shape[0]}')
    csv_res_df.to_json('./data/res_csv/sft/sft-my.json', orient='records', force_ascii=False)


if __name__ == '__main__':
    # make_pt_dataset()
    make_sft_dataset()
