import logging
from logging import getLogger
import re
import textwrap
import os
import json
from dataclasses import asdict
from .components import LightParameters, TrainingState
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def plot_training_state(
    state: TrainingState,
    class_names: Optional[list[str]] = None,
):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("default")

    num_classes = len(state.current_metrics.IoU)
    if class_names is None:
        # 保持类名从 Class_1 开始
        class_names = [f"Class_{i+1}" for i in range(num_classes)]

    # 【升级】使用 GridSpec 创建 3行2列 的高级混合看板，高度调整为 18 以容纳更多图表
    fig = plt.figure(figsize=(16, 18), dpi=100)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1, 1, 1.2]
    )  # 稍微给底部的混淆矩阵多一点空间

    # 顶部大标题，展示核心全局指标
    title_str = (
        f"Training State Dashboard [Epoch {state.current_epoch}]\n"
        f"Current: mIoU={state.current_metrics.mIoU:.4f} | mPrecision={state.current_metrics.mPrecision:.4f} | mRecall={state.current_metrics.mRecall:.4f}\n"
        f"Best Hist: mIoU={state.best_metrics.mIoU:.4f} | mPrecision={state.best_metrics.mPrecision:.4f} | mRecall={state.best_metrics.mRecall:.4f}"
    )
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.98, va="top")

    # --- 1. [左上]：Current Epoch Batch Loss ---
    ax_loss = fig.add_subplot(gs[0, 0])
    losses = state.train_loss
    if isinstance(losses, list) and len(losses) > 0:
        ax_loss.plot(
            range(1, len(losses) + 1),
            losses,
            color="#E64B35",
            alpha=0.8,
            linewidth=1.5,
            label="Batch Loss",
        )
        if len(losses) > 10:
            window = max(2, len(losses) // 10)
            smooth_loss = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax_loss.plot(
                range(window, len(losses) + 1),
                smooth_loss,
                color="#7E2F23",
                linewidth=2.5,
                label=f"Moving Avg (w={window})",
            )
        ax_loss.set_title(
            f"Epoch Loss Curve ({len(losses)} epochs)",
            fontsize=12,
            fontweight="bold",
        )
        ax_loss.set_xlabel("Train Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
    else:
        ax_loss.text(
            0.5, 0.5, "No Loss Data Available", ha="center", va="center", fontsize=12
        )

    ax_miou = fig.add_subplot(gs[0, 1])
    train_mious = state.train_miou
    test_mious = state.test_miou

    if isinstance(train_mious, list) and len(train_mious) > 0:
        epochs_range = range(1, len(train_mious) + 1)
        # 画训练集 mIoU 曲线
        ax_miou.plot(
            epochs_range,
            train_mious,
            color="#00A087",
            marker="o",
            markersize=4,
            linewidth=2,
            label="Train mIoU",
        )
        # 画测试集 mIoU 曲线（如果有的话）
        if isinstance(test_mious, list) and len(test_mious) > 0:
            # 防止两列表长度不一致报错，取对应的范围
            ax_miou.plot(
                range(1, len(test_mious) + 1),
                test_mious,
                color="#3C5488",
                marker="s",
                markersize=4,
                linewidth=2,
                label="Test mIoU",
            )
        ax_miou.set_title(
            "mIoU History Curve (Over Epochs)", fontsize=12, fontweight="bold"
        )
        ax_miou.set_xlabel("Epochs")
        ax_miou.set_ylabel("mIoU Score")
        ax_miou.set_ylim(-0.02, 1.02)
        ax_miou.legend(loc="lower right")
    else:
        ax_miou.text(
            0.5, 0.5, "No mIoU History Data", ha="center", va="center", fontsize=12
        )

    # --- 3. [中间整行拉宽]：Per-Class IoU Comparison (Current vs Best) ---
    ax_bar = fig.add_subplot(gs[1, :])  # gs[1, :] 表示占据第2行的整行
    cur_iou = state.current_metrics.IoU
    best_iou = state.best_metrics.IoU

    x = np.arange(len(class_names))
    width = 0.35

    ax_bar.bar(
        x - width / 2,
        cur_iou,
        width,
        label=f"Epoch {state.current_epoch} IoU",
        color="#4DBBD5",
        edgecolor="white",
    )
    if best_iou and len(best_iou) == num_classes:
        ax_bar.bar(
            x + width / 2,
            best_iou,
            width,
            label="Best History IoU",
            color="#91D1C2",
            edgecolor="white",
        )

    ax_bar.set_title("IoU Per Class: Current vs Best", fontsize=12, fontweight="bold")
    ax_bar.set_xticks(x)
    # 因为拉宽了，类名的排布会非常好看，不易重叠
    ax_bar.set_xticklabels(class_names, rotation=15, ha="right")
    ax_bar.set_ylabel("IoU Score")
    ax_bar.set_ylim(-0.02, 1.02)
    ax_bar.legend(loc="upper right")

    # --- 4. [左下]：Current Confusion Matrix ---
    ax_hm_cur = fig.add_subplot(gs[2, 0])
    hist = np.array(state.hist_matrix)
    if hist.size > 0 and hist.sum() > 0:
        norm_hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-6)
        sns.heatmap(
            norm_hist,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax_hm_cur,
            cbar=False,
        )
        ax_hm_cur.set_title(
            f"Epoch {state.current_epoch} Confusion Matrix (Recall)",
            fontsize=12,
            fontweight="bold",
        )
        ax_hm_cur.set_xlabel("Predicted")
        ax_hm_cur.set_ylabel("True")
    else:
        ax_hm_cur.text(
            0.5, 0.5, "No Current Matrix Data", ha="center", va="center", fontsize=12
        )

    # --- 5. [右下]：Best Confusion Matrix ---
    ax_hm_best = fig.add_subplot(gs[2, 1])
    best_hist = np.array(state.best_hist_matrix)
    if best_hist.size > 0 and best_hist.sum() > 0:
        norm_best_hist = best_hist / (best_hist.sum(axis=1, keepdims=True) + 1e-6)
        sns.heatmap(
            norm_best_hist,
            annot=True,
            fmt=".2f",
            cmap="GnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax_hm_best,
            cbar=False,
        )
        ax_hm_best.set_title(
            "Best History Confusion Matrix (Recall)", fontsize=12, fontweight="bold"
        )
        ax_hm_best.set_xlabel("Predicted")
        ax_hm_best.set_ylabel("True")
    else:
        ax_hm_best.text(
            0.5, 0.5, "No Best Matrix Data", ha="center", va="center", fontsize=12
        )

    # 调整布局，适配 3 行的高度和顶部大标题
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    save_path = os.path.join(state.log_folder, "training_state.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def get_warmup_exponential_lambda(warmup_epochs, start_factor, eta_min, base_lr, gamma):
    """
    Args:
        warmup_epochs: 预热的轮数
        start_factor: 预热起步的学习率缩放倍数
        eta_min: 保底的最小学习率（防止跌到0）
        base_lr: 优化器的初始学习率
        gamma: 指数衰减率（通常在 0.90 到 0.99 之间，越小下降越快）
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup 阶段：线性爬升
            return start_factor + (1.0 - start_factor) * (epoch / warmup_epochs)
        else:
            # 负指数衰减阶段（带 eta_min 保底）
            active_epochs = epoch - warmup_epochs
            # 计算超出 eta_min 之上的那部分学习率如何随指数衰减
            current_lr = eta_min + (base_lr - eta_min) * (gamma**active_epochs)
            # 返回缩放系数
            return current_lr / base_lr

    return lr_lambda
