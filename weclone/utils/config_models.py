from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass


class StrEnum(str, Enum):
    """
    Pydantic-friendly string enum base class
    Supports direct string comparison, e.g.: `if platform == PlatformType.CHAT`
    Also supports string literal comparison, e.g.: `if platform == "chat"`
    """

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None


class BaseConfigModel(BaseModel):
    """Base configuration model with default extra='allow'"""

    model_config = {"extra": "allow"}


class PlatformType(StrEnum):
    """Data source platform"""

    CHAT = "chat"
    TELEGRAM = "telegram"


class LanguageType(StrEnum):
    """Data language"""

    ZH = "zh"
    EN = "en"


class DataModality(StrEnum):
    """Data modality"""

    TEXT = "text"
    IMAGE = "image"
    STICKER = "sticker"
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


class CommonArgs(BaseConfigModel):
    """NOTE that all parameters here will be parsed by `HfArgumentParser`. Non-HfArgumentParser parameters should be placed in make_dataset_args."""

    model_name_or_path: str = Field(...)
    adapter_name_or_path: Optional[str] = Field(None, description="Also as output_dir of train_sft_args")
    template: str = Field(..., description="model template")
    default_system: str = Field(..., description="default system prompt")
    finetuning_type: FinetuningType = Field(FinetuningType.LORA)
    media_dir: str = Field("dataset/media")
    image_max_pixels: int = Field(409920, description="used in llama-factory, 409920 represents 720P")
    enable_thinking: bool = Field(False, description="used in llama-factory")
    trust_remote_code: bool = Field(True, description="used in huggingface")


class CliArgs(BaseModel):
    model_config = {"extra": "forbid"}
    full_log: bool = Field(False)
    log_level: str = Field("INFO", description="DEBUG, INFO, WARNING, ERROR, CRITICAL")


class LLMCleanConfig(BaseConfigModel):
    accept_score: int = Field(
        2,
        description="Acceptable LLM scoring threshold: 1 (worst) to 5 (best). Data scoring below this threshold will not be used for training.",
    )
    enable_thinking: bool = Field(False, description="used in llama-factory")


class CleanDatasetConfig(BaseConfigModel):
    enable_clean: bool = False
    clean_strategy: CleanStrategy = CleanStrategy.LLM
    llm: LLMCleanConfig = LLMCleanConfig(accept_score=2, enable_thinking=False)


class VisionApiConfig(BaseConfigModel):
    """Vision API specific configuration"""

    enable: bool = Field(default=False, description="Whether to enable Vision API for image recognition")
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None
    max_workers: Optional[int] = None


class TelegramArgs(BaseModel):
    model_config = {"extra": "forbid"}
    my_id: str = Field(default="user1234567890", description="Your own telegram id")


class MakeDatasetArgs(BaseConfigModel):
    model_config = {"extra": "forbid"}

    platform: PlatformType = Field(..., description="Data source platform")
    telegram_args: Optional[TelegramArgs] = None
    language: LanguageType = Field(LanguageType.ZH, description="Common language used in chat")
    include_type: List[DataModality] = Field([DataModality.TEXT], description="Types of data to include")
    max_image_num: int = Field(2, description="Maximum number of images per single data entry")
    blocked_words: List[str] = Field([], description="List of blocked words")
    add_time: bool = Field(False, description="Whether to add time to the dataset")
    single_combine_strategy: CombineStrategy = Field(
        CombineStrategy.TIME_WINDOW,
        description="Strategy for combining single person's messages into a single sentence",
    )
    qa_match_strategy: CombineStrategy = Field(
        CombineStrategy.TIME_WINDOW, description="Strategy for forming QA pairs"
    )
    single_combine_time_window: int = Field(
        2, description="Time window for combining single person's messages (minutes)"
    )
    qa_match_time_window: int = Field(5, description="Time window for forming QA pairs (minutes)")
    combine_msg_max_length: int = Field(2048, description="Maximum length of combined messages")
    messages_max_length: int = Field(
        2048, description="Maximum character count for messages, used with cutoff_len"
    )
    prompt_with_history: bool = Field(
        False, description="Whether to include conversation history in prompt, invalid for multimodal data"
    )
    clean_dataset: CleanDatasetConfig = Field(CleanDatasetConfig(), description="Data cleaning configuration")
    online_llm_clear: bool = Field(False)
    base_url: Optional[str] = Field(None, description="Base URL for online LLM")
    llm_api_key: Optional[str] = Field(None, description="API key for online LLM")
    model_name: Optional[str] = Field(
        None, description="Model name for online LLM, recommend using larger parameter models"
    )
    clean_batch_size: int = Field(10, description="Batch size for data cleaning")
    vision_api: VisionApiConfig = Field(VisionApiConfig())


class TrainSftArgs(BaseConfigModel):
    stage: str = Field("sft", description="Training stage")
    dataset: str = Field(..., description="Dataset name")
    dataset_dir: str = Field("./dataset/res_csv/sft", description="Dataset directory")
    freeze_multi_modal_projector: bool = Field(
        False, description="Whether to freeze multimodal projector during MLLM training"
    )
    use_fast_tokenizer: bool = Field(True, description="Whether to use fast tokenizer")
    lora_target: str = Field(..., description="LoRA target modules")
    lora_rank: int = Field(4, description="LoRA rank")
    lora_dropout: float = Field(0.25, description="LoRA dropout")
    weight_decay: float = Field(0.1, description="Weight decay")
    overwrite_cache: bool = Field(True, description="Whether to overwrite cache")
    per_device_train_batch_size: int = Field(4, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Gradient accumulation steps")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler type")
    cutoff_len: int = Field(4096, description="Cutoff length")
    logging_steps: int = Field(10, description="Logging steps")
    save_steps: int = Field(100, description="Model save steps")
    learning_rate: float = Field(1e-4, description="Learning rate")
    warmup_ratio: float = Field(0.1, description="Warmup ratio")
    num_train_epochs: int = Field(2, description="Number of training epochs")
    plot_loss: bool = Field(True, description="Whether to plot loss curve")
    fp16: bool = Field(True, description="Whether to use fp16")
    flash_attn: str = Field("fa2", description="Flash Attention type")
    preprocessing_num_workers: int = Field(16, description="Number of preprocessing worker processes")
    dataloader_num_workers: int = Field(4, description="Number of dataloader worker processes")
    deepspeed: Optional[str] = Field(
        None, description="DeepSpeed configuration file path for multi-GPU training"
    )
    do_train: bool = Field(True)


class InferArgs(BaseConfigModel):
    repetition_penalty: float = Field(1.2, description="Repetition penalty")
    temperature: float = Field(..., description="Temperature")
    top_p: float = Field(..., description="Top-p sampling")
    max_length: int = Field(..., description="Maximum generation length")


class VllmArgs(BaseConfigModel):
    gpu_memory_utilization: float = Field(default=0.9, description="vllm GPU memory utilization")


class TestModelArgs(BaseConfigModel):
    test_data_path: str = Field(default="dataset/eval/test_data-en.json", description="Test data path")


class CommonMethods:
    def _parse_dataset_name(self) -> str:
        """Parse and process dataset name"""
        if hasattr(self, "include_type") and "image" in getattr(self, "include_type", []):
            return getattr(self, "dataset", "") + "-vl"
        return getattr(self, "dataset", "")


class WcConfig(BaseModel):
    model_config = {"extra": "forbid"}

    version: str = Field(..., description="Configuration file version")
    common_args: CommonArgs = Field(..., description="Common parameters")
    cli_args: CliArgs = Field(..., description="Command line arguments")
    make_dataset_args: MakeDatasetArgs = Field(..., description="Dataset processing parameters")
    train_sft_args: TrainSftArgs = Field(..., description="SFT fine-tuning parameters")
    infer_args: InferArgs = Field(..., description="Inference parameters")
    vllm_args: VllmArgs = Field(VllmArgs())
    test_model_args: TestModelArgs = Field(TestModelArgs())


class WCInferConfig(CommonArgs, InferArgs):
    """Final configuration model for Web Demo"""

    pass


class WCTrainSftConfig(CommonArgs, TrainSftArgs, CommonMethods):
    """Final configuration model for SFT training"""

    # Training output directory, converted from adapter_name_or_path
    output_dir: Optional[str] = Field(None)
    dataset: str = Field(..., description="Dataset name")

    @model_validator(mode="after")
    def process_config(self):
        adapter_name_value = getattr(self, "adapter_name_or_path", None)

        if adapter_name_value:
            self.output_dir = adapter_name_value

        self.dataset = self._parse_dataset_name()
        # Always remove adapter_name_or_path field after processing
        if hasattr(self, "adapter_name_or_path"):
            delattr(self, "adapter_name_or_path")
        if hasattr(self, "include_type"):
            delattr(self, "include_type")

        return self


class WCMakeDatasetConfig(CommonArgs, MakeDatasetArgs, CommonMethods):
    """Final configuration model for creating datasets"""

    model_config = {"extra": "allow"}  # Explicitly set to allow

    dataset: str = Field(..., description="Dataset name")
    dataset_dir: str = Field("./dataset/res_csv/sft", description="Dataset directory")
    cutoff_len: int = Field(4096, description="Cutoff length")

    @model_validator(mode="after")
    def process_config(self):
        # Validate Telegram configuration
        if self.platform == PlatformType.TELEGRAM:
            if self.telegram_args is None or self.telegram_args.my_id == "user1234567890":
                logger.error(
                    "When using the Telegram platform, please set a valid `telegram_args.my_id`. The `from_id` in `result.json` for the messages you send represents your user ID."
                )
                exit(1)

        self.dataset = self._parse_dataset_name()

        return self
