"""日志和可视化工具"""
import os
import re
import sys
import logging
import matplotlib.pyplot as plt
from datetime import datetime


def setup_logger(log_dir="logs", is_main_process=True):
    """
    设置logger，同时输出到控制台和文件，并捕获所有错误信息
    
    Args:
        log_dir: 日志文件存储目录
        is_main_process: 是否为主进程（分布式训练中只有主进程输出到控制台）
        
    Returns:
        logger对象, log_file路径
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{current_time}.log")
    
    # 创建logger
    logger = logging.getLogger("train_logger")
    
    # 动态拼接logger名称：在默认名称后追加从CHECKPOINT_PATH_TO中
    # "/mnt/chenlb/xray_moe/results/" 之后的路径片段，并将 "/" 替换为 "_"
    try:
        from configs import config as _cfg  # 延迟导入，避免循环依赖
        checkpoint_path = getattr(_cfg, "CHECKPOINT_PATH_TO", "")
        if isinstance(checkpoint_path, str) and checkpoint_path:
            boundary = "results/"
            if boundary in checkpoint_path:
                suffix_part = checkpoint_path.split(boundary, 1)[1]
            else:
                suffix_part = checkpoint_path
            suffix_part = suffix_part.strip("/").replace("/", "_")
            if suffix_part:
                logger.name = f"train_logger_{suffix_part}"
    except Exception:
        # 配置不可用或解析失败时，保持默认名称
        pass
    logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以捕获所有信息
    
    # 清除已存在的处理器（避免重复）
    logger.handlers.clear()
    
    # 关键：禁止向上传播到根logger，避免在分布式/笔记本环境中重复输出
    logger.propagate = False
    
    # 文件处理器（所有进程都写入日志文件）
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 设置详细格式（包含日志级别）
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    # 添加文件处理器
    logger.addHandler(file_handler)
    
    # 控制台处理器（只有主进程输出到控制台）
    if is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"日志文件: {log_file}")
    
    # 设置未捕获异常的处理器
    def handle_exception(exc_type, exc_value, exc_traceback):
        """处理未捕获的异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许KeyboardInterrupt正常终止程序
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # 记录异常到日志文件
        logger.critical("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    return logger, log_file


def log_metrics(logger, epoch, train_loss, test_loss, result):
    """
    记录训练和测试的评估指标
    
    Args:
        logger: 日志记录器
        epoch: 当前轮次
        train_loss: 训练损失
        test_loss: 测试损失
        result: 包含评估指标的字典
    """
    if epoch is not None:
        logger.info(f"Epoch: {epoch}")
    
    if train_loss is not None:
        logger.info(f"Train Loss: {train_loss:.4f}")
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # 记录其他指标
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        logger.info(f"  {sub_key}: {sub_value:.4f}")


def plot_length_distribution(distribution, title):
    """
    绘制长度分布图
    
    Args:
        distribution: 长度分布字典或列表
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(distribution, dict):
        lengths = list(distribution.keys())
        counts = list(distribution.values())
    else:
        lengths = list(range(len(distribution)))
        counts = distribution
    
    plt.bar(lengths, counts)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def clean_report_mimic_cxr(report):
    """
    清洗MIMIC-CXR报告文本
    
    Args:
        report: 原始报告文本
        
    Returns:
        清洗后的报告文本
    """
    # 清理报告
    report_cleaner = lambda t: (
        t.replace("\n", " ")
        .replace("__", "_")
        .replace("  ", " ")
        .replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )
    
    # 清理句子
    sent_cleaner = lambda t: re.sub(
        "[.,?;*!%^&_+():-\[\]{}]",
        "",
        t.replace('"', "")
        .replace("/", "")
        .replace("\\", "")
        .replace("'", "")
        .strip()
        .lower(),
    )
    
    # 处理
    tokens = [
        sent_cleaner(sent)
        for sent in report_cleaner(report)
        if sent_cleaner(sent) != []
    ]
    
    report = " . ".join(tokens) + " ."
    return report


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visual_parameters(modules, parameters):
    """
    可视化模型参数分布
    
    Args:
        modules: 模块列表
        parameters: 参数列表
    """
    logger = logging.getLogger("train_logger")
    logger.info("\n模型参数统计:")
    logger.info("=" * 60)
    
    total_params = sum(parameters)
    
    for module, params in zip(modules, parameters):
        percentage = (params / total_params) * 100 if total_params > 0 else 0
        logger.info(f"{module:30s}: {params:12,d} ({percentage:5.2f}%)")
    
    logger.info("=" * 60)
    logger.info(f"{'总参数':30s}: {total_params:12,d}")
    logger.info("=" * 60)

