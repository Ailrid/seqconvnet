import logging
from logging import getLogger
import re
import textwrap
import os
import json
from dataclasses import asdict
from .components import LightParameters

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


def save_light_params(
    path: str,
    light_params: LightParameters,
):
    """
    将模型、数据集和环境配置序列化并保存到指定文件夹下的 params.json 文件中
    """
    os.makedirs(path, exist_ok=True)

    combined_dict = {
        "light_params": asdict(light_params),
    }

    file_path = os.path.join(path, "params.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_dict, f, indent=4, ensure_ascii=False)


def confirm_light_params(
    path: str,
    light_params: LightParameters,
):
    """
    基于结构安全白名单，核验关键的模型架构和空间几何参数。
    允许动态调整学习率、Epochs、Batch Size 和文件夹路径。
    """
    file_path = os.path.join(path, "params.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The historical parameter configuration file was not found in the folder: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        # 直接定位到内部的 light_params 字典
        saved_config = json.load(f).get("light_params", {})

    current_config = asdict(light_params)

    # 只有写入这个字典的字段，才会被严格比对
    required_checks = {
        "model_params": ["model_type", "d_model", "nhead", "num_layers", "dropout"],
        "dataset_params": [
            "num_classes",
            "voxel_params",
            "input_size",
        ],  # 纳入 input_size 防御特征图变形
    }

    mismatches = []
    matches = []

    for section_name, checked_keys in required_checks.items():
        saved_section = saved_config.get(section_name, {})
        current_section = current_config.get(section_name, {})

        for key in checked_keys:
            v_saved = saved_section.get(key)
            v_current = current_section.get(key)

            # Python 的 != 会自动递归比较嵌套的 voxel_params 字典
            if v_saved != v_current:
                mismatches.append(
                    f"  ➔ [{section_name}] -> property '{key}':\n"
                    f"      Historical saved values: {v_saved}\n"
                    f"      Current input value: {v_current}"
                )
            else:
                matches.append(
                    f"  ➔ [{section_name}] -> property '{key}':\n"
                    f"      Historical saved values: {v_saved}\n"
                    f"      Current input value: {v_current}\n"
                )

    if mismatches:
        error_title = f"\nParameter Mismatch! The current running configuration is inconsistent with the historically saved configuration."
        error_details = "\n".join(mismatches)
        raise ValueError(
            f"{error_title}\n{error_details}\nPlease check your configuration file or clean up the experiment folder."
        )

    return "".join(matches)
