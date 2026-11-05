import json
import os
from typing import cast

import pytest

# Import common functions from test_full_pipe
from tests.test_full_pipe import (
    get_config_files,
    isolated_config,
    load_config_with_path,
    print_test_header,
    run_cli_command,
    test_logger,
)
from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("config_file", get_config_files())
def test_PII_make_dataset(config_file, tmp_path, monkeypatch):
    """Test PII data make-dataset functionality"""
    print_test_header("PII make-dataset", config_file)
    config_file = isolated_config(config_file, tmp_path, monkeypatch, "test_PII")
    config = cast(WCMakeDatasetConfig, load_config_with_path(config_file, "make_dataset"))

    # Run make-dataset command
    result = run_cli_command(["make-dataset"], config_file)
    assert result.returncode == 0, f"make-dataset command execution failed for config {config_file}"

    # Print all user messages from the dataset file with PII warning
    sft_file_path = os.path.join(config.dataset_dir, "sft-my.json")
    if os.path.exists(sft_file_path):
        logger.warning("⚠️  WARNING: The following content contains unfiltered PII (Personally Identifiable Information):")
        logger.warning("=" * 80)

        with open(sft_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            if 'messages' in entry:
                for message in entry['messages']:
                    if message.get('role') == 'user':
                        logger.warning(f"User content: {message.get('content', '')}")

        logger.warning("=" * 80)
        logger.warning("⚠️  END OF UNFILTERED PII CONTENT")

    test_logger.info(f"✅ PII make-dataset test passed for config {config_file}")
