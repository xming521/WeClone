import re
from typing import List, Optional, cast

import torch
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.misc import get_device_count
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams

from weclone.utils.config import load_config
from weclone.utils.config_models import VllmArgs
from weclone.utils.log import logger

# from vllm.entrypoints.openai.tool_parsers import xLAMToolParser

# NOTE: the V1 LLM engine writing style was used.


def extract_json_from_text(text: str) -> str:
    """Extract JSON content from text, supporting JSON blocks in markdown format."""
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def vllm_infer(
    inputs: List[str],
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    default_system: Optional[str] = None,
    enable_thinking: bool = False,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    guided_decoding_class: Optional[type[BaseModel]] = None,
    bad_words: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
) -> tuple[List[RequestOutput] | List[Optional[BaseModel]], List[int]]:
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Returns:
        tuple: (results, failed_indices) where failed_indices contains indices of failed JSON parsing
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    wc_vllm_args = cast(VllmArgs, load_config("vllm"))
    model_args, data_args, _, generating_args = get_infer_args(
        {
            "model_name_or_path": model_name_or_path,
            "adapter_name_or_path": adapter_name_or_path,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "template": template,
            "cutoff_len": cutoff_len,
            "max_samples": max_samples,
            "preprocessing_num_workers": 16,
            "vllm_config": vllm_config,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "enable_thinking": enable_thinking,
        }
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    if guided_decoding_class:
        json_schema = guided_decoding_class.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema, disable_any_whitespace=True)

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k or -1,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        bad_words=bad_words,
        guided_decoding=guided_decoding_params if guided_decoding_class else None,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "enable_prefix_caching": True,
        "guided_decoding_backend": "guidance",
        "guided_decoding_disable_any_whitespace": True,
    }

    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    wc_vllm_dict = {k: v for k, v in wc_vllm_args.model_dump().items() if v is not None}
    engine_args.update(wc_vllm_dict)

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    messages_list = [[{"role": "user", "content": text}] for text in inputs]

    llm = LLM(**engine_args)

    results = llm.chat(
        messages_list,
        sampling_params,
        lora_request=lora_request,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )  # type: ignore

    del llm
    torch.cuda.empty_cache()

    failed_indexs = []
    if guided_decoding_class:
        # TODO better json decode  https://github.com/vllm-project/vllm/commit/1d0ae26c8544fd5a62e171e30c2dcc2973a23bc8#diff-3b27790a2ce97bc50cdd5476f7b0057da682ed0d1ec8426a7b76c5e21454e57d
        parsed_results = []
        for idx, result in enumerate(results):
            try:
                json_text = extract_json_from_text(result.outputs[0].text)
                parsed_result = guided_decoding_class.model_validate_json(json_text)
                parsed_results.append(parsed_result)
            except Exception as e:
                # Note that the failed_indexs is the sequential index ID, not the original input ID.
                logger.warning(
                    f"Failed to parse JSON from result at sequence index {idx}: {result.outputs[0].text[:100]}..., error: {e}"
                )
                failed_indexs.append(idx)
                parsed_results.append(None)
        results = parsed_results

    return results, failed_indexs
