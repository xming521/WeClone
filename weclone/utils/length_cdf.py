# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import fire
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from tqdm import tqdm

from weclone.utils.log import logger


def calculate_token_length(
    text: str,
    model_name_or_path: str = "./models/Qwen3-32B-AWQ",
    template: str = "qwen3",
) -> int:
    """Calculate the token length of the specified text

    Args:
        text: Text to calculate token length for
        model_name_or_path: Model path
        template: Template name

    Returns:
        Token length of the text
    """
    logger.info(f"Calculating text token length: {text[:50]}...")

    model_args, data_args, _, _, _ = get_train_args(
        {
            "stage": "sft",
            "model_name_or_path": model_name_or_path,
            "template": template,
            "dataset": "chat-sft",
            "output_dir": "dummy_dir",
            "do_train": True,
        }
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # Directly use tokenizer to encode text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_length = len(tokens)

    logger.info(f"Text token length: {token_length}")
    return token_length


def length_cdf(
    model_name_or_path: str = "./Qwen2.5-7B-Instruct",
    dataset: str = "chat-sft",
    dataset_dir: str = "./dataset/res_csv/sft",
    media_dir: str = "./dataset/media",
    template: str = "qwen",
    interval: int = 256,
    image_max_pixels: int = 768 * 768,
):
    r"""Calculate the distribution of the input lengths in the dataset.

    Usage: export CUDA_VISIBLE_DEVICES=0
    python length_cdf.py --model_name_or_path path_to_model --dataset alpaca_en_demo --template default
    """
    logger.info("Starting cutoff_len calculation......")

    model_args, data_args, training_args, _, _ = get_train_args(
        {
            "stage": "sft",
            "model_name_or_path": model_name_or_path,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "template": template,
            "cutoff_len": 1_000_000,
            "preprocessing_num_workers": 16,
            "output_dir": "dummy_dir",
            "media_dir": media_dir,
            "image_max_pixels": int(image_max_pixels),
            "overwrite_cache": True,
            "do_train": True,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    template_obj = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)  # type: ignore
    trainset = get_dataset(template_obj, model_args, data_args, training_args, "sft", **tokenizer_module)[
        "train_dataset"
    ]  # type: ignore
    total_num = len(trainset)  # type: ignore
    length_dict = defaultdict(int)
    for sample in tqdm(trainset["input_ids"], desc="Collecting lengths"):  # type: ignore
        length_dict[len(sample) // interval * interval] += 1

    length_tuples = list(length_dict.items())
    length_tuples.sort()
    count_accu, prob_accu = 0, 0
    logger.info(" cutoff_len configuration suggestions:")
    logger.warning("For multimodal tasks, please ensure cutoff_len is set to the maximum data length")
    for length, count in length_tuples:
        count_accu += count
        prob_accu += count / total_num * 100
        logger.info(f"{count_accu:d} ({prob_accu:.2f}%) samples have length < {length + interval}.")


if __name__ == "__main__":
    fire.Fire(length_cdf)
