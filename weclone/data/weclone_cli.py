import logging
import re
import argparse
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directories(base_dir='weclone_improved'):
    dirs = ['data_pipeline', 'llm_training', 'interfaces', 'utils', 'models', 'logs']
    for d in dirs:
        path = Path(base_dir) / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def anonymize_chat(chat_text: str) -> str:
    
    chat_text = re.sub(r'\b\d{10,13}\b', '[PHONE]', chat_text)
    chat_text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', chat_text)
    chat_text = re.sub(r'\b[A-Z][a-z]+\b', '[NAME]', chat_text)
    return chat_text


def main():
    parser = argparse.ArgumentParser(description="WeClone CLI Tool")
    parser.add_argument('--init', action='store_true', help="Initialize project directories")
    parser.add_argument('--anonymize', type=str, help="Path to chat file to anonymize")
    args = parser.parse_args()

    if args.init:
        create_directories()
        logger.info("Project structure initialized.")

    if args.anonymize:
        path = Path(args.anonymize)
        if path.exists():
            original = path.read_text(encoding='utf-8')
            clean = anonymize_chat(original)
            output_path = path.with_stem(path.stem + '_anonymized')
            output_path.write_text(clean, encoding='utf-8')
            logger.info(f"Anonymized chat saved to {output_path}")
        else:
            logger.error("File not found.")

if __name__ == '__main__':
    main()
