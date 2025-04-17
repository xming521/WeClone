from loguru import logger
import sys

# 移除默认的处理器
logger.remove()

logger.add(
    sys.stderr,
    format="<green>[Weclone]</green> <level>{level}</level> | <level>{message}</level>",
    colorize=True,
)
