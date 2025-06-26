import logging
import sys
import time
from functools import wraps

from loguru import logger

logger.remove()

# 添加WeClone专用的sink
logger.add(
    sys.stderr,
    format="<green><b>[WeClone]</b></green> <level>{level.name[0]}</level> | <level>{time:HH:mm:ss}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO",
)


# 桥接标准logging到loguru
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


# 配置标准logging使用loguru
# 你可以在这里修改 InterceptHandler 的级别：
intercept_handler = InterceptHandler(level=logging.INFO)  # 只显示 INFO 及以上级别
logging.basicConfig(handlers=[intercept_handler], level=0, force=True)


# 便捷函数：动态设置 intercepted logging 的级别
def set_intercepted_logging_level(level):
    """
    设置被intercepted的标准logging的日志级别

    Args:
        level: 日志级别，可以是：
            - logging.DEBUG (10)
            - logging.INFO (20)
            - logging.WARNING (30)
            - logging.ERROR (40)
            - logging.CRITICAL (50)
            或者字符串: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    intercept_handler.setLevel(level)
    logger.info(f"Intercepted logging level set to: {logging.getLevelName(level)}")


logger.add(
    "logs/weclone.log",  # 日志文件路径
    rotation="1 day",  # 每天轮换一个新的日志文件
    retention="7 days",  # 保留最近7天的日志文件
    compression="zip",  # 压缩旧的日志文件
    level="DEBUG",  # 文件日志级别
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",  # 日志格式
    encoding="utf-8",  # 文件编码
    enqueue=True,  # 异步写入，避免阻塞
)


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
