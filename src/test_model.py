import json
import logging
import os
import sys
import openai
sys.path.append(os.getcwd())
import tqdm
from src.template import default_prompt
from tqdm import tqdm


config = {
    'default_prompt': default_prompt,
    'model': 'gpt-3.5-turbo',
    'history_len': 15,
}

config = type('Config', (object,), config)()

openai.api_key = '''sk-test'''
openai.api_base = "http://127.0.0.1:8000/v1"


def handler_text(content: str, history: [], config):

    messages = [{"role": "system", "content": f'{config.default_prompt}'}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    try:
        response = openai.ChatCompletion.create(model=config.model,
                                                messages=messages,
                                                max_tokens=50)
    except openai.APIError as e:
        history.pop()
        return 'AI接口出错,请重试\n' + str(e)

    resp = str(response.choices[0].message.content)
    resp = resp.replace('\n ', '')
    history.append({"role": "assistant", "content": resp})
    return resp


def main():
    test_list = json.loads(open('data/test_data.json').read())['questions']
    res = []
    for questions in tqdm(test_list, desc=' Testing...'):
        history = []
        for q in questions:
            answer = handler_text(q, history=history, config=config)
        res.append(history)

    res_file = open('test_result-my.txt', 'w')
    for r in res:
        for i in r:
            res_file.write(i['content'] + '\n')
        res_file.write('\n')


if __name__ == '__main__':
    main()
