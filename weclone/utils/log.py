from loguru import logger
import sys

logger.remove()

logger.add(
    sys.stderr,
    format="<green><b>[WeClone]</b></green> <level>{level.name[0]}</level> | <level>{time:HH:mm:ss}</level> | <level>{message}</level>",
    colorize=True,
)
