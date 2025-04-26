import click

# 只在需要时导入，避免启动时加载大型库
# from weclone.data.qa_generator import DataProcessor
# from weclone.eval.web_demo import main as web_demo_main
# from weclone.eval.eval_model import main as evaluate_main
# from weclone.server.api_service import main as server_main
# from weclone.train.train_sft import main as train_sft_main


@click.group()
def cli():
    """WeClone: 从聊天记录创造数字分身的一站式解决方案"""
    pass


@cli.command(help="处理聊天记录CSV文件，生成问答对数据集。")
def qa_generator():
    """处理聊天记录CSV文件，生成问答对数据集。"""
    from weclone.data.qa_generator import DataProcessor

    processor = DataProcessor()
    processor.main()


@cli.command("train-sft", help="使用准备好的数据集对模型进行微调。")
def train_sft():
    """使用准备好的数据集对模型进行微调。"""
    from weclone.train.train_sft import main as train_sft_main

    train_sft_main()


@cli.command("web-demo", help="启动 Web UI 与微调后的模型进行交互测试。")  # 命令名修改为 web-demo
def web_demo():
    """启动 Web UI 与微调后的模型进行交互测试。"""
    from weclone.eval.web_demo import main as web_demo_main

    web_demo_main()


@cli.command("evaluate", help="使用常见问题测试微调后模型的效果。")
def evaluate():
    """使用常见问题测试微调后模型的效果。"""
    from weclone.eval.eval_model import main as evaluate_main

    evaluate_main()


@cli.command("server", help="启动API服务，提供模型推理接口。")
def server():
    """启动API服务，提供模型推理接口。"""
    from weclone.server.api_service import main as server_main

    server_main()


if __name__ == "__main__":
    cli()
