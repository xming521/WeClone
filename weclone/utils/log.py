import sys
from functools import wraps

from loguru import logger

logger.remove()

logger.add(
    sys.stderr,
    format="<green><b>[WeClone]</b></green> <level>{level.name[0]}</level> | <level>{time:HH:mm:ss}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO",
)

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
