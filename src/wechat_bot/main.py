import logging
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
sys.path.append(os.getcwd())
import openai
import requests

import itchat
from handler.text import handler_text
from itchat import utils
from itchat.content import *
from src.template import default_prompt
import logging

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger('main')

config = {
    'default_prompt': default_prompt,
    'model': 'gpt-3.5-turbo',
    'history_len': 15,
}


config = type('Config', (object,), config)()


def stop_program(signal, frame):
    log.info('WeChatbot Closing Save some data')
    itchat.dump_login_status()
    sys.exit(0)


signal.signal(signal.SIGTERM, stop_program)


class WeChatGPT:

    def __init__(self):
        itchat.auto_login(enableCmdQR=2, hotReload=True, statusStorageDir='./cookie.bin')

        self.history = {}
        self.prompts = {}
        openai.api_key = '''sk-test'''
        openai.api_base = "http://127.0.0.1:8000/v1"

        log.info("init successful!")

    def handler_history(self, msg):
        self.history.setdefault(msg.user.userName, [])
        history = self.history[msg.user.userName]
        need_remove_len = len(history) - config.history_len
        if need_remove_len > 0:
            for i in range(need_remove_len):
                # 必须出一对 
                history.pop(0)
                history.pop(0)
        return history

    def reply(self, msg):
        if time.time() - msg.CreateTime > 5:
            return None
        res = handler_text(content=msg.text, history=self.handler_history(msg), config=config)
        res = res.split('，')
        res[-1] = res[-1].replace('。', '')
        if res[0] == '':
            res[0] = '机器人他无语了'
        for r in res:
            msg.user.send(r)
            time.sleep(2.2)

    def run(self):
        @itchat.msg_register(FRIENDS)
        def add_friend(msg):
            """自动同意好友"""
            root = ET.fromstring(msg.content)
            ticket = root.get('ticket')
            # itchat.accept_friend(msg.user.userName, ticket)

        @itchat.msg_register(TEXT)
        def friend(msg):
            """处理私聊消息"""
            log.info(f"{msg.user.NickName}: {msg.text}")
            self.reply(msg)

        @itchat.msg_register(TEXT, isGroupChat=True)
        def groups(msg):
            """处理群聊消息"""
            if msg.isAt:
                self.reply(msg)

        itchat.run(debug=True)


if __name__ == "__main__":

    try:
        weChatGPT = WeChatGPT()
        weChatGPT.run()
    except KeyboardInterrupt:
        log.info("bye!")
