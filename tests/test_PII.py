import os
import shutil
import subprocess
import sys
from typing import cast

import pytest

# Import common functions from test_full_pipe
from tests.test_full_pipe import (
    DATASET_CSV_DIR,
    PROJECT_ROOT_DIR,
    get_config_files,
    load_config_with_path,
    print_test_header,
    run_cli_command,
    setup_data_environment,
    test_logger,
)
from weclone.utils.config import load_config
from weclone.utils.config_models import DataModality, WCMakeDatasetConfig
from weclone.utils.log import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup paths
TESTS_DIR = os.path.dirname(__file__)
TEST_DATA_PII_DIR = os.path.join(TESTS_DIR, "tests_data", "test_PII")

@pytest.mark.parametrize("config_file", get_config_files())
def test_PII_make_dataset(config_file):
    """Test PII data make-dataset functionality"""
    print_test_header("PII make-dataset", config_file)
    
    setup_data_environment("test_PII")
    
    # Load config and handle images if needed
    config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config_with_path(config_file, "make_dataset"))

    # Run make-dataset command
    result = run_cli_command(["make-dataset"], config_file)
    assert result.returncode == 0, f"make-dataset command execution failed for config {config_file}"

    # Print all user messages from the dataset file with PII warning
    import json
    sft_file_path = os.path.join(PROJECT_ROOT_DIR, "dataset", "res_csv", "sft", "sft-my.json")
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

if __name__ == "__main__":
    # If running directly, run tests for all configs
    for config_file in get_config_files():
        test_PII_make_dataset(config_file) 
