from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass


class StrEnum(str, Enum):
    """
    Pydantic-friendly string enum base class
    Supports direct string comparison, e.g.: `if platform == PlatformType.WECHAT`
    Also supports string literal comparison, e.g.: `if platform == "wechat"`
    """

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None


class PlatformType(StrEnum):
    """Data source platform"""

    WECHAT = "wechat"
    # QQ = "qq"
    # TELEGRAM = "telegram"


class DataModality(StrEnum):
    """Data modality"""

    TEXT = "text"
    IMAGE = "image"
    # AUDIO = "audio"
    # VIDEO = "video"


class CombineStrategy(StrEnum):
    """Combination strategy"""

    TIME_WINDOW = "time_window"


class CleanStrategy(StrEnum):
    """Data cleaning strategy"""

    LLM = "llm"


class FinetuningType(StrEnum):
    """Finetuning type"""

    LORA = "lora"
    # FULL = "full"
    # FREEZE = "freeze"


class CommonArgs(BaseModel):
    model_config = {"extra": "ignore"}

    model_name_or_path: str = Field(...)
    adapter_name_or_path: str = Field("./model_output", description="Also as output_dir of train_sft_args")
    template: str = Field(..., description="model template")
    default_system: str = Field(..., description="default system prompt")
    finetuning_type: FinetuningType = Field(FinetuningType.LORA)
    media_dir: str = Field("dataset/media")
    image_max_pixels: int = Field(409920, description="used in llama-factory, 409920代表720P")
    enable_thinking: bool = Field(False, description="used in llama-factory")
    trust_remote_code: bool = Field(True, description="used in huggingface")


class CliArgs(BaseModel):
    full_log: bool = Field(False)


class LLMCleanConfig(BaseModel):
    accept_score: int = Field(
        2,
        description="Acceptable LLM scoring threshold: 1 (worst) to 5 (best). Data scoring below this threshold will not be used for training.",
    )


class CleanDatasetConfig(BaseModel):
    enable_clean: bool = False
    clean_strategy: CleanStrategy = CleanStrategy.LLM
    llm: LLMCleanConfig = LLMCleanConfig(accept_score=2)


class VisionApiConfig(BaseModel):
    """Vision API specific configuration"""

    enable: bool = Field(default=False, description="是否启用Vision API进行图像识别")
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None
    max_workers: Optional[int] = None


class MakeDatasetArgs(BaseModel):
    platform: PlatformType = Field(..., description="Data source platform")
    include_type: List[DataModality] = Field([DataModality.TEXT], description="包含的数据类型")
    max_image_num: int = Field(2, description="单条数据最大图片数量")
    blocked_words: List[str] = Field([], description="禁用词列表")
    single_combine_strategy: CombineStrategy = Field(
        CombineStrategy.TIME_WINDOW, description="单人组成单句策略"
    )
    qa_match_strategy: CombineStrategy = Field(CombineStrategy.TIME_WINDOW, description="组成QA策略")
    single_combine_time_window: int = Field(2, description="单人组成单句时间窗口（分钟）")
    qa_match_time_window: int = Field(5, description="组成QA时间窗口（分钟）")
    combine_msg_max_length: int = Field(2048, description="组合后消息最大长度")
    messages_max_length: int = Field(2048, description="messages最长字符数量, 配合cutoff_len 使用")
    prompt_with_history: bool = Field(False, description="是否在prompt中包含历史对话, 多模态数据此配置无效")
    clean_dataset: CleanDatasetConfig = Field(CleanDatasetConfig(), description="数据清洗配置")
    online_llm_clear: bool = Field(False)
    base_url: Optional[str] = Field(None, description="在线LLM的base_url")
    llm_api_key: Optional[str] = Field(None, description="在线LLM的api_key")
    model_name: Optional[str] = Field(None, description="在线LLM的模型名称, 建议使用参数较大的模型")
    clean_batch_size: int = Field(10, description="数据清洗批次大小")
    vision_api: VisionApiConfig = Field(VisionApiConfig())


class TrainSftArgs(BaseModel):
    model_config = {"extra": "ignore"}

    stage: str = Field("sft", description="训练阶段")
    dataset: str = Field(..., description="数据集名称")
    dataset_dir: str = Field("./dataset/res_csv/sft", description="数据集目录")
    freeze_multi_modal_projector: bool = Field(False, description="MLLM 训练时是否冻结多模态投影器")
    use_fast_tokenizer: bool = Field(True, description="是否使用快速分词器")
    lora_target: str = Field(..., description="LoRA目标模块")
    lora_rank: int = Field(4, description="LoRA秩")
    lora_dropout: float = Field(0.25, description="LoRA dropout")
    weight_decay: float = Field(0.1, description="权重衰减")
    overwrite_cache: bool = Field(True, description="是否覆盖缓存")
    per_device_train_batch_size: int = Field(4, description="每设备训练批次大小")
    gradient_accumulation_steps: int = Field(8, description="梯度累积步数")
    lr_scheduler_type: str = Field("cosine", description="学习率调度器类型")
    cutoff_len: int = Field(4096, description="截断长度")
    logging_steps: int = Field(10, description="日志记录步数")
    save_steps: int = Field(100, description="模型保存步数")
    learning_rate: float = Field(1e-4, description="学习率")
    warmup_ratio: float = Field(0.1, description="预热比例")
    num_train_epochs: int = Field(2, description="训练轮数")
    plot_loss: bool = Field(True, description="是否绘制损失曲线")
    fp16: bool = Field(True, description="是否使用fp16")
    flash_attn: str = Field("fa2", description="Flash Attention类型")
    preprocessing_num_workers: int = Field(16, description="预处理工作进程数")
    dataloader_num_workers: int = Field(4, description="数据加载工作进程数")
    deepspeed: Optional[str] = Field(None, description="DeepSpeed配置文件路径, 用于多卡训练")
    do_train: bool = Field(True)


class InferArgs(BaseModel):
    repetition_penalty: float = Field(1.2, description="重复惩罚")
    temperature: float = Field(..., description="温度")
    top_p: float = Field(..., description="Top-p采样")
    max_length: int = Field(..., description="最大生成长度")


class VllmArgs(BaseModel):
    gpu_memory_utilization: float = Field(default=0.9, description="vllm GPU内存利用率")


class TestModelArgs(BaseModel):
    test_data_path: str = Field(default="dataset/test_data.json", description="测试数据路径")


class WcConfig(BaseModel):
    version: str = Field(..., description="配置文件版本")
    common_args: CommonArgs = Field(..., description="通用参数")
    cli_args: CliArgs = Field(..., description="命令行参数")
    make_dataset_args: MakeDatasetArgs = Field(..., description="数据处理参数")
    train_sft_args: TrainSftArgs = Field(..., description="SFT微调参数")
    infer_args: InferArgs = Field(..., description="推理参数")
    vllm_args: VllmArgs = Field(VllmArgs())
    test_model_args: TestModelArgs = Field(TestModelArgs())


class WCInferConfig(CommonArgs, InferArgs):
    """用于Web Demo的最终配置模型"""

    model_config = {"extra": "ignore"}


class WCTrainSftConfig(CommonArgs, TrainSftArgs):
    """用于SFT训练的最终配置模型"""

    model_config = {"extra": "ignore"}

    # 训练输出目录，从adapter_name_or_path转换而来
    output_dir: Optional[str] = Field(None)

    @model_validator(mode="after")
    def process_config(self):
        # 保存需要的值
        adapter_name_value = getattr(self, "adapter_name_or_path", None)

        # 进行业务逻辑处理
        if self.dataset == "wechat-sft":
            self.dataset = "chat-sft"
        if adapter_name_value:
            self.output_dir = adapter_name_value

        try:
            delattr(self, "adapter_name_or_path")
        except AttributeError:
            pass

        return self


class WCMakeDatasetConfig(CommonArgs, MakeDatasetArgs):
    """用于创建数据集的最终配置模型"""

    model_config = {"extra": "ignore"}

    dataset: str = Field(..., description="数据集名称")
    dataset_dir: str = Field("./dataset/res_csv/sft", description="数据集目录")
    cutoff_len: int = Field(4096, description="截断长度")

    @model_validator(mode="after")
    def process_config(self):
        if self.dataset == "wechat-sft":
            self.dataset = "chat-sft"
        return self
