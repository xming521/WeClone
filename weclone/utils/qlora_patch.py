import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

__original_from_pretrained = AutoModelForCausalLM.from_pretrained

def apply_qlora_patch():
    """
    Monkey-patch AutoModelForCausalLM.from_pretrained，
    将所有模型加载统一应用 4-bit QLoRA 配置。

    配置参数说明：
    - load_in_4bit=True：启用 4-bit 量化，显存占用大幅降低
    - bnb_4bit_quant_type="nf4"：选择 NF4 量化策略，兼顾精度与模型大小
    - bnb_4bit_compute_dtype=torch.float16：推理/训练时使用半精度进行计算，进一步节省显存
    - bnb_4bit_use_double_quant=True：双重量化，提升量化效果和模型性能
    """
    def _patched_from_pretrained(*args, **kwargs):
        qlora_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        # Remove conflicting arg if present
        kwargs.pop("load_in_4bit", None)
        # Inject QLoRA config
        kwargs["quantization_config"] = qlora_config
        return __original_from_pretrained(*args, **kwargs)

    AutoModelForCausalLM.from_pretrained = _patched_from_pretrained