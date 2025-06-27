import logging
import os
import sys
import time
from functools import wraps

from loguru import logger

logger.remove()

env_log_level = os.getenv("WC_LOG_LEVEL")
# 初始化基本日志配置，稍后会被 configure_log_level_from_config 重新配置
logger.add(
    sys.stderr,
    format="<green><b>[WeClone]</b></green> <level>{level.name[0]}</level> | <level>{time:HH:mm:ss}</level> | <level>{message}</level>",
    colorize=True,
    level=env_log_level.upper() if env_log_level else "INFO",
)


class InterceptHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)

    def emit(self, record):
        # 检查日志级别，只处理指定级别及以上的日志
        if record.levelno < self.level:
            return

        timestamp = time.strftime("%H:%M:%S")
        level_color = "\033[36m" if record.levelno >= logging.INFO else "\033[0m"
        reset_color = "\033[0m"
        message = f"[{record.name}] | {level_color}{record.levelname[0]}{reset_color} | {timestamp} | {record.getMessage()}"
        print(message, file=sys.stderr)


# 桥接标准logging到loguru
intercept_handler = InterceptHandler(level=logging.INFO)
logging.basicConfig(handlers=[intercept_handler], level=0, force=True)


def capture_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_sink_buffer = []

        def list_sink(message):
            log_sink_buffer.append(message.record["message"])

        sink_id = logger.add(list_sink, format="{message}", level="INFO")

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        class OutputTeeToGlobalLog:
            def __init__(self, original_stream, log_method):
                self.original_stream = original_stream
                self.log_method = log_method
                self.current_line_content = ""  # Represents the current state of the line to be logged

            def write(self, data_chunk):
                self.original_stream.write(data_chunk)  # Pass through to console

                if data_chunk.endswith("\\r") and "\\n" not in data_chunk:
                    self.current_line_content = data_chunk[:-1]  # Store without the trailing \\r
                    return

                full_buffer = self.current_line_content + data_chunk
                lines_to_process = full_buffer.split("\\n")

                for i in range(len(lines_to_process) - 1):
                    line = lines_to_process[i]
                    final_content_of_line = line
                    last_cr = line.rfind("\\r")
                    if last_cr != -1:
                        final_content_of_line = line[last_cr + 1 :]

                    escaped_log = final_content_of_line.replace("{", "{{").replace("}", "}}")
                    if final_content_of_line.strip() or line:
                        self.log_method(escaped_log, raw=True)

                self.current_line_content = lines_to_process[-1]

            def flush(self):
                self.original_stream.flush()
                if self.current_line_content:
                    final_content_of_line = self.current_line_content
                    last_cr = self.current_line_content.rfind("\\r")
                    if last_cr != -1:
                        final_content_of_line = self.current_line_content[last_cr + 1 :]

                    escaped_log = final_content_of_line.replace("{", "{{").replace("}", "}}")
                    if final_content_of_line.strip() or self.current_line_content:
                        self.log_method(escaped_log, raw=True)
                    self.current_line_content = ""

        sys.stdout = OutputTeeToGlobalLog(original_stdout, logger.opt(raw=True).info)
        sys.stderr = OutputTeeToGlobalLog(original_stderr, logger.opt(raw=True).error)

        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logger.remove(sink_id)

    return wrapper


def configure_log_level_from_config():
    """
    从配置文件中读取日志等级并设置完整的日志配置
    需要在配置加载后调用
    """
    log_level = "INFO"  # 默认值

    try:
        from weclone.utils.config import load_config

        cli_config = load_config(arg_type="cli_args")
        log_level = getattr(cli_config, "log_level", "INFO")
    except Exception as e:
        logger.warning(f"无法从配置加载日志等级，使用默认INFO级别: {e}")

    logger.remove()

    logger.add(
        sys.stderr,
        format="<green><b>[WeClone]</b></green> <level>{level.name[0]}</level> | <level>{time:HH:mm:ss}</level> | <level>{message}</level>",
        colorize=True,
        level=log_level.upper(),
    )

    logger.add(
        "logs/weclone.log",
        rotation="1 day",
        retention="7 days",
        compression="zip",
        level="DEBUG",  # 文件日志始终保持DEBUG级别，便于调试
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
        enqueue=True,
    )

    intercept_handler.setLevel(log_level.upper())

    logger.info(f"日志等级已设置为: {log_level.upper()}")
