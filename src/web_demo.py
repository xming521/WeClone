import sys
from llmtuner import create_web_demo
from template import template_register


def dict_to_argv(d):
    argv = []
    for k, v in d.items():
        argv.append('--' + k)
        if v is not None:
            argv.append(str(v))
    return argv


# 使用示例
config = {
    'model_name_or_path': './chatglm3-6b',
    'adapter_name_or_path': './model_output',
    'template': 'chatglm3-weclone',
    'finetuning_type': 'lora',
    'repetition_penalty': 1.2,
    'temperature': 0.5,
    'max_length': 50,
    'top_p': 0.65
}

sys.argv += dict_to_argv(config)

template_register()


def main():
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


# python LLaMA-Factory/src/web_demo.py \
#     --model_name_or_path ./chatglm3-6b \
#     --adapter_name_or_path ./model_output \
#     --template chatglm3 \
#     --finetuning_type lora\
#     --repetition_penalty 1.2\
#     --temperature 0.5\
#     --max_length 50\
#     --top_p 0.65

if __name__ == "__main__":
    main()
