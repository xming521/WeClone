# WEClone 测试指南

本目录包含WEClone项目的测试文件，用于确保项目各个组件正常工作。

## 测试文件说明

- `test_weclone_pipeline.py`: 全流程测试，按顺序测试数据生成、训练、API服务和模型评估
- `test_qa_generator.py`: 测试QA生成器功能


## 运行全流程测试

要运行完整的测试流程，请执行以下命令：

```bash
# 在项目根目录下执行
python -m tests.test_weclone_pipeline
```

## 测试流程说明

全流程测试按照以下顺序测试项目的主要组件：

1. **数据生成**：测试 `weclone/data/qa_generator.py` 模块，模拟微信聊天记录的处理和QA对的生成
2. **模型训练**：测试 `weclone/train/train_sft.py` 模块，模拟使用生成的数据进行模型的SFT训练
3. **API服务**：测试 `weclone/server/api_service.py` 模块，模拟启动API服务
4. **模型评估**：测试 `weclone/eval/test_model.py` 模块，模拟对训练后的模型进行评估

## 注意事项

- 测试使用Python的unittest框架和mock库，模拟各个组件的运行环境和依赖
- 测试不会修改实际的数据文件或模型文件，所有操作都在临时目录中进行
- 要运行单独的测试方法，可以使用以下命令：

```bash
# 例如，只运行QA生成器测试
python -m unittest tests.test_weclone_pipeline.TestWeclonePipeline.test_qa_generator
```

