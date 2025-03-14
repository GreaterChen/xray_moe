import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd


# ------ Helper Functions ------
def data_to_device(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, tuple):
        data = tuple(data_to_device(item, device) for item in data)
    elif isinstance(data, list):
        data = list(data_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        data = dict((k, data_to_device(v, device)) for k, v in data.items())
    # else:
    # raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')

    return data


def data_concatenate(iterable_data, dim=0):
    data = iterable_data[0]  # can be a list / tuple / dict / tensor
    if isinstance(data, torch.Tensor):
        return torch.cat([*iterable_data], dim=dim)
    elif isinstance(data, tuple):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in range(num_cols):
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return tuple(return_data)
    elif isinstance(data, list):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in range(num_cols):
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return list(return_data)
    elif isinstance(data, dict):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in data.keys():
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return dict((k, return_data[i]) for i, k in enumerate(data.keys()))
    else:
        raise TypeError("Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.")


def data_distributor(model, source):
    if isinstance(source, torch.Tensor):
        output = model(source)
    elif isinstance(source, tuple) or isinstance(source, list):
        output = model(*source)
    elif isinstance(source, dict):
        output = model(**source)
    else:
        raise TypeError("Unsupported DataType! Try List/Tuple!")
    return output


def args_to_kwargs(
    args, kwargs_list=None
):  # This function helps distribute input to corresponding arguments in Torch models
    if kwargs_list != None:
        if isinstance(args, dict):  # Nothing to do here
            return args
        else:  # args is a list or tuple or single element
            if isinstance(args, torch.Tensor):  # single element
                args = [args]
            assert len(args) == len(kwargs_list)
            return dict(zip(kwargs_list, args))
    else:  # Nothing to do here
        return args


def prepare_batch_data(args, batch, data_loader, device, findings=True, history=True, label=True, bbox=True):
    """准备批次数据，对整个batch进行tokenization

    Args:
        batch: 输入的批次数据
        data_loader: 数据加载器
        device: 计算设备

    Returns:
        source_data: 源数据字典
        target_data: 目标数据字典
    """
    source = {}
    target = {}
    # 处理图像数据
    if "image" in batch:
        batch["image"] = data_to_device(batch["image"], device)
        source['image'] = batch['image']

    # 根据参数确定需要处理的文本字段
    text_fields = []
    if findings and "findings" in batch:
        text_fields.append("findings")
    if history and "history" in batch:
        text_fields.append("history")

    # 对整个batch的文本进行tokenization
    text_fields = ["findings", "history"]
    for field in text_fields:
        if field in batch:
            # 收集batch中的所有文本
            texts = batch[field]

            if field == "history":
                max_len = args.max_len_history
            elif field == "findings":
                max_len = args.max_len_findings

            # 对整个batch进行tokenization
            encoded = data_loader.dataset.tokenizer(
                texts,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(
                device
            )  # [batch_size, max_len]
            batch[field] = encoded

            source[field] = batch[field]
            target[field] = batch[field]

    # 将label移到device上
    if label and "label" in batch:
        batch["label"] = data_to_device(batch["label"], device)
        source["label"] = batch["label"]
        target["label"] = batch["label"]

    # 将targets移到device上
    if bbox and "bbox_targets" in batch:
        batch["bbox_targets"] = data_to_device(batch["bbox_targets"], device)
        source["bbox_targets"] = batch["bbox_targets"]

    return source, target, None


# ------ Core Functions ------
def train(
    args,
    data_loader,
    model,
    optimizer,
    criterion,
    num_epochs,
    current_epoch,
    scheduler=None,
    device="cpu",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
    scaler=None,
):
    model.train()
    running_loss = 0

    prog_bar = tqdm(data_loader)
    for i, batch in enumerate(prog_bar):
        # 准备批次数据
        if args.phase == "TRAIN_DETECTION":
            source, target, _ = prepare_batch_data(args, batch, data_loader, device, findings=False, history=False, label=False, bbox=True)
        elif args.phase == "PRETRAIN_VIT":
            source, target, _ = prepare_batch_data(args, batch, data_loader, device, findings=True, history=False, label=True, bbox=True)
        else:
            pass
        # 转换为kwargs格式
        source = args_to_kwargs(source)
        target = args_to_kwargs(target)

        source["phase"] = args.phase
        source["mode"] = "train"
        source['current_epoch'] = current_epoch
        source['total_epochs'] = num_epochs

        scheduler.step(cur_epoch=current_epoch, cur_step=i)
        current_lr = optimizer.param_groups[0]["lr"]
        if scaler != None:
            with torch.cuda.amp.autocast():
                output = data_distributor(model, source)
                output = args_to_kwargs(output)
                if args.phase == "TRAIN_DETECTION":
                    # 汇总所有损失项
                    loss = sum(loss for loss in output.values())
                elif args.phase == "PRETRAIN_VIT":
                    loss = output['ltc_loss'] + output['cls_loss']
                else:
                    pass


            running_loss += loss.item()
            prog_bar.set_description(f"Loss: {running_loss/(i+1)} | LR: {current_lr}")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)
            loss = criterion(output, target)

            running_loss += loss.item()
            prog_bar.set_description(
                "Loss: {}".format(round(running_loss / (i + 1), 8))
            )

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

    return running_loss / len(data_loader)

def test(
    args,
    data_loader,
    model,
    logger,
    mode="val",
    metric_ftns=None,
    criterion=None,
    device="cpu",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
    epoch=None,
):
    model.eval()
    running_loss = 0

    # 初始化存储列表
    findings_gts_list = []
    findings_preds_list = []
    impression_gts_list = []
    impression_preds_list = []
    image_paths_list = []
    splits_list = []
    labels_list = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, batch in enumerate(prog_bar):
            # 收集元数据
            image_paths_list.extend(batch["image_path"])
            splits_list.extend(batch["split"])
            labels_list.extend(batch["label"].cpu().numpy().tolist())

            # 收集ground truth
            findings_gts_list.extend([gt for gt in batch["gts"][0]])
            impression_gts_list.extend([gt for gt in batch["gts"][1]])

            # 准备批次数据
            if args.phase == "TRAIN_DETECTION":
                source, target, _ = prepare_batch_data(args, batch, data_loader, device, findings=False, history=False, label=False, bbox=True)
            elif args.phase == "PRETRAIN_VIT":
                source, target, _ = prepare_batch_data(args, batch, data_loader, device, findings=False, history=False, label=True, bbox=True)
            else:
                pass

            # 转换为kwargs格式
            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)

            source["phase"] = args.phase
            source["mode"] = mode

            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])

            # 计算损失
            if criterion is not None:
                loss = torch.tensor(0.0)
                running_loss += loss.item()
            prog_bar.set_description("Loss: {}".format(running_loss / (i + 1)))

        # 创建结果数据字典
        results_data = {
            "image_path": image_paths_list,
            "split": splits_list,
            "findings_gt": findings_gts_list,
            "findings_pred": findings_preds_list,
            "labels": labels_list,
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            * len(findings_gts_list),
        }

        # 计算评估指标
        findings_met = metric_ftns(
            {i: [gt] for i, gt in enumerate(findings_gts_list)},
            {i: [re] for i, re in enumerate(findings_preds_list)},
        )

        # 创建结果目录
        results_dir = os.path.join(args.checkpoint_path_to, "test_results")
        os.makedirs(results_dir, exist_ok=True)

        # 将结果转换为DataFrame并保存
        results_df = pd.DataFrame(results_data)

        # 保存为CSV文件，添加epoch信息
        epoch_str = str(epoch) if epoch is not None else "TEST"
        csv_filename = f"{mode}_results_epoch_{epoch_str}.csv"
        results_df.to_csv(os.path.join(results_dir, csv_filename), index=False)
        logger.info(f"结果已保存到CSV文件: {csv_filename}")

        # 计算并保存评估指标
        metrics_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "epoch": epoch_str,
            "loss": running_loss / len(data_loader),
        }

        # 添加findings指标
        for metric_name, value in findings_met.items():
            metrics_data[f"findings_{metric_name}"] = value

        # 保存评估指标，添加epoch信息
        metrics_df = pd.DataFrame([metrics_data])
        metrics_filename = f"{mode}_metrics_epoch_{epoch_str}.csv"
        metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
        logger.info(f"评估指标已保存到CSV文件: {metrics_filename}")

        # 返回结果
        result = {
            "findings_met": findings_met,
            "loss": running_loss / len(data_loader),
            "results_df": results_df,
            "metrics_df": metrics_df,
        }

    return running_loss / len(data_loader), result

def infer_bert(
    args,
    data_loader,
    model,
    num_epochs,
    current_epoch,
    device="cuda",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
    scaler=None,
):
    model.eval()
    running_loss = 0

    prog_bar = tqdm(data_loader)
    for i, batch in enumerate(prog_bar):
        source, target, _ = prepare_batch_data(args, batch, data_loader, device, findings=True, history=False, label=True, bbox=False)
        # 转换为kwargs格式
        source = args_to_kwargs(source)
        target = args_to_kwargs(target)

        source["phase"] = args.phase
        source["mode"] = "train"
        source['current_epoch'] = current_epoch
        source['total_epochs'] = num_epochs

        output = data_distributor(model, source)
        output = args_to_kwargs(output)

    save_path = os.path.join(args.checkpoint_path_to, "negative_pool", "pool.npy")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    model.negative_pool.save(save_path)

    return output

def save(path, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(
        {
            # --- Model Statistics ---
            "epoch": epoch,
            "stats": stats,
            # --- Model Parameters ---
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer != None else None
            ),
            # 'scheduler_state_dict': scheduler.state_dict() if scheduler != None else None,
        },
        path,
    )


def load(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    # --- Model Statistics ---
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    # --- 提取detector部分的参数 ---
    if "model_state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint["model_state_dict"]
        
        # 检查是否是MOE模型的检查点
        is_moe_checkpoint = any(key.startswith("object_detector.") for key in checkpoint_state_dict.keys())
        
        if is_moe_checkpoint:
            print("检测到MOE模型检查点，正在提取目标检测器参数...")
            # 创建一个新的state_dict，只包含detector部分
            detector_state_dict = {}
            
            # 遍历checkpoint中的所有键
            for key, value in checkpoint_state_dict.items():
                # 如果键以"object_detector."开头，则提取
                if key.startswith("object_detector."):
                    # 去掉"object_detector."前缀
                    new_key = key[len("object_detector."):]
                    detector_state_dict[new_key] = value
            
            # 加载提取后的state_dict到模型
            missing_keys, unexpected_keys = model.load_state_dict(
                detector_state_dict, strict=False
            )
        else:
            # 如果不是MOE检查点，直接尝试加载
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint_state_dict, strict=False
            )
    else:
        print("检查点中没有找到模型状态字典！")
        return epoch, stats

    # 打印未加载和多余的参数信息
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    if optimizer != None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except:  # Input optimizer doesn't fit the checkpoint one --> should be ignored
            print("Cannot load the optimizer")
    
    return epoch, stats


def log_metrics(logger, epoch, train_loss, test_loss, result):
    """记录训练和测试的评估指标

    Args:
        logger: 日志记录器
        epoch: 当前轮次(可选,如果是None则不输出)
        train_loss: 训练损失(可选,如果是None则不输出)
        test_loss: 测试损失
        result: 包含metrics_df的结果字典
    """
    if epoch is not None:
        logger.info(f"epoch: {epoch}")
    if train_loss is not None:
        logger.info(f"train_loss: {train_loss}")
    logger.info(f"test_loss: {test_loss:.4f}")

    # 输出评估指标
    metrics_df = result["metrics_df"]
    for index, row in metrics_df.iterrows():
        for column in metrics_df.columns:
            value = row[column]
            if isinstance(value, (int, float)):
                logger.info(f"{column}: {value:.4f}")
            else:
                logger.info(f"{column}: {value}")


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visual_parameters(modules, parameters):
    # 数据
    total_parameters = sum(parameters)

    # 计算占比
    percentages = [param / total_parameters * 100 for param in parameters]

    # 绘制饼状图
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=modules, autopct="%1.1f%%", startangle=140)
    plt.title("Parameter Distribution Among Modules")
    plt.savefig(
        "/home/chenlb/xray_report_generation/results/parameter_distribution.png"
    )


# 绘制直方图
def plot_length_distribution(distribution, title):
    # 准备数据
    lengths = list(distribution.keys())  # Token 长度
    counts = list(distribution.values())  # 出现次数

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts)
    plt.title(title)
    plt.xlabel("Token Length")
    plt.ylabel("Count")

    # 计算合适的刻度间隔
    max_length = max(lengths)
    min_length = min(lengths)
    step = max(1, (max_length - min_length) // 5)  # 最多显示约10个刻度

    # 设置x轴刻度
    plt.xticks(range(min_length, max_length + 1, step))

    plt.tight_layout()
    plt.savefig(
        f"/home/chenlb/xray_report_generation/results/{title.replace(' ', '_')}.png"
    )


def setup_logger(log_dir="logs"):
    """
    设置logger，同时输出到控制台和文件

    Args:
        log_dir: 日志文件存储目录
    Returns:
        logger: 配置好的logger对象
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名（使用当前时间）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{current_time}.log")

    # 创建logger对象
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def clean_report_mimic_cxr(report):
    report_cleaner = (
        lambda t: t.replace("\n", " ")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
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
    tokens = [
        sent_cleaner(sent)
        for sent in report_cleaner(report)
        if sent_cleaner(sent) != []
    ]
    report = " . ".join(tokens) + " ."
    return report


def analyze_results_from_csv(csv_path, metric_ftns=None):
    """从CSV文件中分析结果并计算评估指标，包括findings、impression以及它们的组合

    Args:
        csv_path: CSV文件路径
        metric_ftns: 计算指标的函数，默认为None

    Returns:
        dict: 包含findings、impression和combined指标的字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查必要的列是否存在
    required_columns = ["findings_gt", "findings_pred"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV文件必须包含以下列: {required_columns}")

    # 准备findings的ground truth和预测结果
    findings_gts = {i: [gt] for i, gt in enumerate(df["findings_gt"])}
    findings_preds = {i: [pred] for i, pred in enumerate(df["findings_pred"])}

    # 计算findings的指标
    findings_metrics = metric_ftns(findings_gts, findings_preds) if metric_ftns else {}

    # 初始化impression和combined的指标为None
    impression_metrics = None
    combined_metrics = None

    # 如果存在impression相关列，计算impression和combined的指标
    if "impression_gt" in df.columns and "impression_pred" in df.columns:
        # 计算impression的指标
        impression_gts = {}
        impression_preds = {}
        idx = 0
        for i in range(len(df)):
            if (
                df["impression_gt"][i].strip() != ""
                and df["impression_pred"][i].strip() != ""
            ):
                impression_gts[idx] = [df["impression_gt"][i]]
                impression_preds[idx] = [df["impression_pred"][i].replace(".", " .")]
                idx += 1
        # impression_gts = {i: [gt] for i, gt in enumerate(df["impression_gt"])}
        # impression_preds = {i: [pred] for i, pred in enumerate(df["impression_pred"])}
        impression_metrics = (
            metric_ftns(impression_gts, impression_preds) if metric_ftns else {}
        )

        # 计算combined (findings + impression)的指标
        combined_gts = {}
        combined_preds = {}
        for i, (f_gt, i_gt, f_pred, i_pred) in enumerate(
            zip(
                df["findings_gt"],
                df["impression_gt"],
                df["findings_pred"],
                df["impression_pred"],
            )
        ):
            # 只有当impression不为空时才组合
            if isinstance(i_gt, str) and len(i_gt.strip()) > 0:
                combined_gts[i] = [f"{f_gt} {i_gt}"]
                combined_preds[i] = [f"{f_pred} {i_pred}"]
            else:
                combined_gts[i] = [f_gt]
                combined_preds[i] = [f_pred]

        combined_metrics = (
            metric_ftns(combined_gts, combined_preds) if metric_ftns else {}
        )

    # 整理结果
    results = {
        "findings_metrics": findings_metrics,
        "impression_metrics": impression_metrics,
        "combined_metrics": combined_metrics,
    }

    return results


def save_generations(
    args,
    data_loader,
    model,
    logger,
    save_dir,
    mode="test",
    device="cpu",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
):
    """保存模型生成的findings和impression结果

    Args:
        args: 配置参数
        data_loader: 数据加载器
        model: 模型
        logger: 日志记录器
        save_dir: 保存结果的目录
        mode: 运行模式，默认为"test"
        device: 计算设备
        kw_src: source关键字参数列表
        kw_tgt: target关键字参数列表
        kw_out: output关键字参数列表
    """
    model.eval()

    # 初始化存储列表
    findings_gts_list = []
    findings_preds_list = []
    impression_gts_list = []
    impression_preds_list = []
    image_paths_list = []
    splits_list = []
    labels_list = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for batch in prog_bar:
            # 收集元数据
            image_paths_list.extend(batch["image_path"])
            splits_list.extend(batch["split"])
            labels_list.extend(batch["label"].cpu().numpy().tolist())

            # 收集ground truth
            findings_gts_list.extend([gt for gt in batch["gts"][0]])
            impression_gts_list.extend([gt for gt in batch["gts"][1]])

            # 准备批次数据
            source, target, _ = prepare_batch_data(args, batch, data_loader, device)

            # 转换为kwargs格式
            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)

            source["mode"] = mode

            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建结果数据字典
    results_data = {
        "image_path": image_paths_list,
        "split": splits_list,
        "findings_gt": findings_gts_list,
        "findings_pred": findings_preds_list,
        "labels": labels_list,
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        * len(findings_gts_list),
    }

    # 将结果转换为DataFrame并保存
    results_df = pd.DataFrame(results_data)

    # 生成文件名并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{mode}_generations_{timestamp}.csv"
    save_path = os.path.join(save_dir, csv_filename)
    results_df.to_csv(save_path, index=False)

    logger.info(f"生成结果已保存到: {save_path}")
    logger.info(f"总共保存了 {len(findings_gts_list)} 条记录")


def test_detection(
    args,
    data_loader,
    model,
    logger,
    mode="val",
    iou_threshold=0.5,
    confidence_threshold=0.5,
    device="cuda",
    epoch=None,
):
    """
    评估目标检测模型性能 - 简化版
    
    参数:
        args: 配置参数
        data_loader: 测试数据加载器
        model: DetectionOnlyFastRCNN模型实例
        logger: 日志记录器
        mode: 评估模式 ("val" 或 "test")
        iou_threshold: 判定为成功检测的IoU阈值
        confidence_threshold: 检测置信度阈值
        device: 计算设备
        epoch: 当前训练轮次
    
    返回:
        float: 平均损失
        dict: 包含评估结果的字典
    """
    model.eval()
    running_loss = 0
    
    # 初始化存储结构
    all_predictions = []
    all_ground_truths = []
    image_paths_list = []
    
    # 按类别存储预测和真值
    num_classes = 29  # 假设有29个区域类别
    class_predictions = {i: [] for i in range(1, num_classes+1)}
    class_ground_truths = {i: [] for i in range(1, num_classes+1)}
    
    # 创建进度条
    prog_bar = tqdm(data_loader, desc=f"{mode} Detection Evaluation")
    
    with torch.no_grad():
        for i, batch in enumerate(prog_bar):
            # 收集图像路径
            image_paths_list.extend(batch["image_path"])
            
            # 准备数据 - 简化处理
            images = batch["image"].to(device)
            targets = []
            
            for target_dict in batch["bbox_targets"]:
                # 将目标数据移动到设备上
                target = {
                    'boxes': target_dict['boxes'].to(device),
                    'labels': target_dict['labels'].to(device),
                    'image_id': target_dict['image_id'].to(device),
                    'area': target_dict['area'].to(device),
                    'iscrowd': target_dict['iscrowd'].to(device)
                }
                targets.append(target)
            
            # 进行前向传播，获取检测结果
            try:
                # 尝试检测模式
                detections = model(images)
            except Exception as e:
                # 如果失败，尝试传入空目标以避免计算损失
                logger.warning(f"检测异常: {e}，尝试传入空目标进行推理")
                detections = model(images, [])
            
            # 计算损失（如果需要）- 这步是可选的
            if targets:
                try:
                    loss_dict = model(images, targets)
                    if isinstance(loss_dict, dict) and all(k.startswith('loss') for k in loss_dict.keys()):
                        batch_loss = sum(loss for loss in loss_dict.values())
                        running_loss += batch_loss.item()
                except Exception as e:
                    logger.warning(f"损失计算异常: {e}")
            
            # 处理检测结果
            for j, (detection, target) in enumerate(zip(detections, targets if targets else [None] * len(detections))):
                # 应用置信度阈值
                keep = detection['scores'] > confidence_threshold
                pred_boxes = detection['boxes'][keep]
                pred_labels = detection['labels'][keep]
                pred_scores = detection['scores'][keep]
                
                # 存储预测结果
                img_pred = {
                    'boxes': pred_boxes.cpu(),
                    'labels': pred_labels.cpu(),
                    'scores': pred_scores.cpu(),
                    'image_id': i * len(images) + j
                }
                all_predictions.append(img_pred)
                
                # 存储真值
                if target is not None:
                    img_gt = {
                        'boxes': target['boxes'].cpu(),
                        'labels': target['labels'].cpu(),
                        'image_id': i * len(images) + j
                    }
                    all_ground_truths.append(img_gt)
                    
                    # 按类别存储预测和真值
                    for class_id in range(1, num_classes + 1):
                        # 提取当前类别的预测
                        class_pred_mask = pred_labels.cpu() == class_id
                        class_predictions[class_id].append({
                            'boxes': pred_boxes.cpu()[class_pred_mask] if class_pred_mask.sum() > 0 else torch.zeros((0, 4)),
                            'scores': pred_scores.cpu()[class_pred_mask] if class_pred_mask.sum() > 0 else torch.zeros(0),
                            'image_id': i * len(images) + j
                        })
                        
                        # 提取当前类别的真值
                        class_gt_mask = target['labels'].cpu() == class_id
                        class_ground_truths[class_id].append({
                            'boxes': target['boxes'].cpu()[class_gt_mask] if class_gt_mask.sum() > 0 else torch.zeros((0, 4)),
                            'image_id': i * len(images) + j
                        })
            
            # 更新进度条
            prog_bar.set_description(
                f"Loss: {running_loss/(i+1):.4f}" if running_loss > 0 else "Evaluating..."
            )
    
    # 计算指标
    logger.info("计算目标检测评估指标...")
    
    # 计算整体mAP
    overall_metrics = calculate_detection_metrics(all_predictions, all_ground_truths, iou_threshold)
    
    # 计算每个类别的指标
    class_metrics = {}
    for class_id in range(1, num_classes + 1):
        class_metrics[class_id] = calculate_class_metrics(
            class_predictions[class_id], 
            class_ground_truths[class_id], 
            iou_threshold
        )
    
    # 计算平均指标
    valid_classes = [c for c in class_metrics.keys() if class_metrics[c]['num_samples'] > 0]
    if valid_classes:
        average_precision = np.mean([class_metrics[c]['AP'] for c in valid_classes])
        average_recall = np.mean([class_metrics[c]['recall'] for c in valid_classes])
        average_f1 = np.mean([class_metrics[c]['f1_score'] for c in valid_classes])
    else:
        average_precision = 0.0
        average_recall = 0.0
        average_f1 = 0.0
    
    # 创建结果目录
    results_dir = os.path.join(args.checkpoint_path_to, "detection_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 整理结果数据
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "epoch": str(epoch) if epoch is not None else "TEST",
        "mAP": average_precision,
        "mRecall": average_recall,
        "mF1": average_f1,
        "loss": running_loss / len(data_loader) if running_loss > 0 else 0.0,
    }
    
    # 添加每个类别的指标
    for class_id, metrics in class_metrics.items():
        metrics_data[f"class_{class_id}_AP"] = metrics['AP']
        metrics_data[f"class_{class_id}_Precision"] = metrics['precision']
        metrics_data[f"class_{class_id}_Recall"] = metrics['recall']
        metrics_data[f"class_{class_id}_F1"] = metrics['f1_score']
    
    # 保存评估指标
    metrics_df = pd.DataFrame([metrics_data])
    epoch_str = str(epoch) if epoch is not None else "TEST"
    metrics_filename = f"{overall_metrics['mAP']}{mode}_detection_metrics_epoch_{epoch_str}.csv"
    metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
    logger.info(f"目标检测评估指标已保存到CSV文件: {metrics_filename}")
    
    # 打印主要指标
    logger.info(f"mAP@{iou_threshold}: {average_precision:.4f}")
    logger.info(f"Mean Recall: {average_recall:.4f}")
    logger.info(f"Mean F1 Score: {average_f1:.4f}")
    
    # 按AP值排序类别
    sorted_classes = sorted([(c, class_metrics[c]['AP']) for c in valid_classes], key=lambda x: x[1], reverse=True)
    
    # 打印表现最好的5个类别
    logger.info("\n表现最好的5个解剖区域:")
    for class_id, ap in sorted_classes[:5]:
        metrics = class_metrics[class_id]
        logger.info(f"区域 {class_id}: AP={ap:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # 打印表现最差的5个类别
    logger.info("\n表现最差的5个解剖区域:")
    for class_id, ap in sorted_classes[-5:]:
        metrics = class_metrics[class_id]
        logger.info(f"区域 {class_id}: AP={ap:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # 返回结果
    result = {
        "overall_metrics": overall_metrics,
        "class_metrics": class_metrics,
        "mAP": average_precision,
        "mRecall": average_recall,
        "mF1": average_f1,
        "loss": running_loss / len(data_loader) if running_loss > 0 else 0.0,
        "metrics_df": metrics_df
    }
    
    return running_loss / len(data_loader) if running_loss > 0 else 0.0, result


def calculate_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    计算目标检测的整体指标
    
    参数:
        predictions: 预测结果列表
        ground_truths: 真值列表
        iou_threshold: IoU阈值
        
    返回:
        dict: 包含mAP等指标的字典
    """
    if not predictions or not ground_truths:
        return {'mAP': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    # 按图像ID和类别整理预测和真值
    pred_by_img_cls = {}
    gt_by_img_cls = {}
    
    for pred in predictions:
        img_id = pred['image_id']
        for i, (box, label, score) in enumerate(zip(pred['boxes'], pred['labels'], pred['scores'])):
            key = (img_id, label.item())
            if key not in pred_by_img_cls:
                pred_by_img_cls[key] = []
            pred_by_img_cls[key].append((box, score))
    
    for gt in ground_truths:
        img_id = gt['image_id']
        for i, (box, label) in enumerate(zip(gt['boxes'], gt['labels'])):
            key = (img_id, label.item())
            if key not in gt_by_img_cls:
                gt_by_img_cls[key] = []
            gt_by_img_cls[key].append(box)
    
    # 计算所有类别的AP
    all_classes = set([k[1] for k in pred_by_img_cls.keys()] + [k[1] for k in gt_by_img_cls.keys()])
    aps = []
    
    for cls in all_classes:
        # 收集此类别的所有预测和真值
        all_preds = []
        all_gt_count = 0
        
        for img_id in set([k[0] for k in pred_by_img_cls.keys()] + [k[0] for k in gt_by_img_cls.keys()]):
            key = (img_id, cls)
            
            # 获取此图像中此类别的预测
            img_preds = pred_by_img_cls.get(key, [])
            img_preds.sort(key=lambda x: x[1], reverse=True)  # 按置信度排序
            
            # 获取此图像中此类别的真值
            img_gts = gt_by_img_cls.get(key, [])
            all_gt_count += len(img_gts)
            
            # 判断每个预测是否为TP
            gt_matched = [False] * len(img_gts)
            
            for pred_box, score in img_preds:
                # 计算与所有真值的IoU
                max_iou = 0
                max_idx = -1
                
                for j, gt_box in enumerate(img_gts):
                    if gt_matched[j]:
                        continue
                    
                    # 计算IoU
                    iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
                
                # 添加预测结果 (score, tp)
                if max_idx >= 0 and max_iou >= iou_threshold:
                    all_preds.append((score.item(), 1))
                    gt_matched[max_idx] = True
                else:
                    all_preds.append((score.item(), 0))
        
        # 计算此类别的AP
        if all_preds:
            # 按置信度排序
            all_preds.sort(reverse=True)
            
            # 计算累积TP和FP
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            
            for score, tp in all_preds:
                if tp == 1:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / max(all_gt_count, 1)
                
                precisions.append(precision)
                recalls.append(recall)
            
            # 计算AP (11点插值法)
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if not recalls or max(recalls) < t:
                    p = 0
                else:
                    # 找到所有大于等于t的recall点
                    p_vals = [precisions[i] for i in range(len(recalls)) if recalls[i] >= t]
                    p = max(p_vals) if p_vals else 0
                ap += p / 11
            
            aps.append(ap)
    
    # 计算平均AP
    mAP = np.mean(aps) if aps else 0.0
    
    # 计算整体精确度、召回率和F1
    total_tp = sum(pred[1] for preds in pred_by_img_cls.values() for pred in preds if pred[1] == 1)
    total_fp = sum(1 for preds in pred_by_img_cls.values() for pred in preds if pred[1] == 0)
    total_gt = sum(len(gts) for gts in gt_by_img_cls.values())
    
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def calculate_class_metrics(class_predictions, class_ground_truths, iou_threshold=0.5):
    """
    计算单个类别的检测指标
    
    参数:
        class_predictions: 此类别的预测列表
        class_ground_truths: 此类别的真值列表
        iou_threshold: IoU阈值
        
    返回:
        dict: 包含AP等指标的字典
    """
    # 计算预测总数和真值总数
    total_preds = sum(len(pred['boxes']) for pred in class_predictions)
    total_gts = sum(len(gt['boxes']) for gt in class_ground_truths)
    
    # 如果没有预测或真值，返回零指标
    if total_preds == 0 or total_gts == 0:
        return {
            'AP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'TP': 0,
            'FP': total_preds,
            'FN': total_gts,
            'num_samples': total_gts
        }
    
    # 初始化TP、FP计数和得分列表
    all_scores = []
    all_tp = []
    
    # 对每张图像进行评估
    for i, (pred, gt) in enumerate(zip(class_predictions, class_ground_truths)):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        # 跳过没有预测或真值的图像
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # 计算所有预测框和真值框之间的IoU
        ious = box_iou(pred_boxes, gt_boxes)
        
        # 对每个预测框，找到最大IoU的真值框
        max_ious, max_idx = ious.max(dim=1)
        
        # 初始化匹配标记
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        # 按置信度排序预测框
        _, sorted_idx = torch.sort(pred_scores, descending=True)
        
        # 判断每个预测是TP还是FP
        for idx in sorted_idx:
            if max_ious[idx] >= iou_threshold:
                gt_idx = max_idx[idx].item()
                if not gt_matched[gt_idx]:
                    # 真阳性
                    all_tp.append(1)
                    gt_matched[gt_idx] = True
                else:
                    # 假阳性(重复检测)
                    all_tp.append(0)
            else:
                # 假阳性(IoU过低)
                all_tp.append(0)
            
            all_scores.append(pred_scores[idx].item())
    
    # 如果没有有效预测，返回零指标
    if not all_scores:
        return {
            'AP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'TP': 0,
            'FP': total_preds,
            'FN': total_gts,
            'num_samples': total_gts
        }
    
    # 按置信度排序
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tp = [all_tp[i] for i in sorted_indices]
    
    # 计算累积TP和FP
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum([1 - tp for tp in all_tp])
    
    # 计算精确度和召回率
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gts
    
    # 计算AP (11点插值法)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if len(recalls) == 0 or max(recalls) < t:
            p = 0
        else:
            p_vals = [precisions[i] for i in range(len(recalls)) if recalls[i] >= t]
            p = max(p_vals) if p_vals else 0
        ap += p / 11
    
    # 计算最终TP、FP、FN和整体指标
    TP = sum(all_tp)
    FP = len(all_tp) - TP
    FN = total_gts - TP
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / total_gts if total_gts > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {
        'AP': ap,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'num_samples': total_gts
    }