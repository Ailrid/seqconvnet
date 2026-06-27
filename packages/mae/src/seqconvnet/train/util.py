from dataclasses import dataclass
import logging
from logging import getLogger
import re
import textwrap

# 颜色与样式转义码定义
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
GRAY = "\x1b[90m"

# 用于清除文件日志中 ANSI 颜色代码的正则表达式
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class ViridConsoleFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_prefix(self, record):
        if record.levelno == logging.INFO:
            return f"{GREEN}{BOLD}[INFO]{RESET} ", 7
        elif record.levelno == logging.WARNING:
            return f"{YELLOW}{BOLD}[WARN]{RESET} ", 7
        elif record.levelno == logging.ERROR:
            if "context" in getattr(record, "msg_type", ""):
                return f"{MAGENTA}{BOLD}✖ [ERR_CTX]{RESET} ", 11
            else:
                return f"{RED}{BOLD}✖ [ERROR]{RESET} ", 9
        return "", 0

    def format(self, record):
        raw_msg = record.getMessage()
        record.message = raw_msg

        # 消除首尾换行与代码缩进污染
        msg_str = textwrap.dedent(raw_msg.strip("\n"))

        prefix, _ = self._get_prefix(record)

        lines = msg_str.splitlines()
        if len(lines) > 1:
            indent = " " * 4
            formatted_lines = [f"{prefix}\n{lines[0]}"]
            for line in lines[1:]:
                formatted_lines.append(f"{indent}{line}")
            return "\n".join(formatted_lines)
        else:
            return f"{prefix}\n{msg_str}"


class ViridFileFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_prefix(self, record):
        if record.levelno == logging.INFO:
            return "[INFO] ", 7
        elif record.levelno == logging.WARNING:
            return "[WARN] ", 7
        elif record.levelno == logging.ERROR:
            if "context" in getattr(record, "msg_type", ""):
                return "✖ [ERR_CTX] ", 11
            else:
                return "✖ [ERROR] ", 9
        return "", 0

    def format(self, record):
        raw_msg = record.getMessage()
        record.message = raw_msg

        # 消除多行日志的换行与环境缩进污染
        msg_str = textwrap.dedent(raw_msg.strip("\n"))

        # 把可能混进文件日志里的 ANSI 颜色代码通通清洗掉！
        msg_str = ANSI_ESCAPE.sub("", msg_str)

        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        time_prefix = f"[{time_str}] "
        level_prefix, _ = self._get_prefix(record)
        full_prefix = f"{time_prefix}{level_prefix}\n"

        lines = msg_str.splitlines()
        if len(lines) > 1:
            indent = " " * 4
            file_lines = [f"{full_prefix}{lines[0]}"]
            for line in lines[1:]:
                file_lines.append(f"{indent}{line}")
            return "\n".join(file_lines)
        else:
            return f"{full_prefix}{msg_str}"


def create_logger(log_file_path: str = "training.txt") -> logging.Logger:
    logger = getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ViridConsoleFormatter())
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(
            log_file_path, mode="a", encoding="utf-8", delay=True
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(ViridFileFormatter())
        logger.addHandler(file_handler)

    return logger
