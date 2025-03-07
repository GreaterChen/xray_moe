import re
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def prepare_batch_data(args, batch, data_loader, device):
    """准备批次数据，对整个batch进行tokenization

    Args:
        batch: 输入的批次数据
        data_loader: 数据加载器
        device: 计算设备

    Returns:
        source_data: 源数据字典
        target_data: 目标数据字典
    """
    # 处理图像数据
    if "image" in batch:
        batch["image"] = data_to_device(batch["image"], device)

    # 对整个batch的文本进行tokenization
    text_fields = ["findings", "impression", "history"]
    for field in text_fields:
        if field in batch:
            # 收集batch中的所有文本
            texts = batch[field]

            if field == "history":
                max_len = args.max_len_history
            elif field == "findings":
                max_len = args.max_len_findings
            elif field == "impression":
                max_len = args.max_len_impression

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
            # encoded.input_ids[:, 0] = data_loader.dataset.tokenizer.bos_token_id
            batch[field] = encoded

    # 将label移到device上
    if "label" in batch:
        batch["label"] = data_to_device(batch["label"], device)

    # 组装source和target
    source = []
    target = []

    for src in data_loader.dataset.sources:
        source.append(batch[src])

    for tgt in data_loader.dataset.targets:
        target.append(batch[tgt])

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
    train_stage=2,
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
        findings_gt = batch["findings"]
        # 准备批次数据
        source, target, _ = prepare_batch_data(args, batch, data_loader, device)

        # 转换为kwargs格式
        source = args_to_kwargs(source, kw_src)
        target = args_to_kwargs(target, kw_tgt)

        source["train_stage"] = train_stage
        source["mode"] = "train"
        source["findings_gt"] = findings_gt

        scheduler.step(cur_epoch=current_epoch, cur_step=i)
        current_lr = optimizer.param_groups[0]["lr"]

        if scaler != None:
            with torch.cuda.amp.autocast():
                output = data_distributor(model, source)
                output = args_to_kwargs(output, kw_out)
                if train_stage == 1:
                    loss = output["loss_lm"]
                elif train_stage == 2:
                    loss, _ = criterion(output, target)
                    loss = loss + output["impression_loss"]
                elif train_stage == 3:
                    loss, _ = criterion(output, target)
                    loss = loss + output["loss_lm"]

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
    train_stage=2,
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
            source, target, _ = prepare_batch_data(args, batch, data_loader, device)

            # 转换为kwargs格式
            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)

            source["train_stage"] = train_stage
            source["mode"] = mode

            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])
            if train_stage != 1:
                impression_preds_list.extend([re for re in output["impression_text"]])

            # 记录日志
            # logger.info(f"findings_preds: {output['findings_text'][0]}")
            # if train_stage == 2:
            #     logger.info(f"impression_preds: {output['impression_text'][0]}")
            # logger.info(f"impression_gt: {batch['gts'][1][0]}")

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

        # 添加impression相关数据(如果是第二阶段)
        if train_stage != 1:
            results_data["impression_gt"] = impression_gts_list
            results_data["impression_pred"] = impression_preds_list

        # 计算评估指标
        findings_met = metric_ftns(
            {i: [gt] for i, gt in enumerate(findings_gts_list)},
            {i: [re] for i, re in enumerate(findings_preds_list)},
        )

        impression_met = None
        if train_stage != 1:
            impression_met = metric_ftns(
                {i: [gt] for i, gt in enumerate(impression_gts_list)},
                {i: [re] for i, re in enumerate(impression_preds_list)},
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
            "train_stage": train_stage,
            "epoch": epoch_str,
            "loss": running_loss / len(data_loader),
        }

        # 添加findings指标
        for metric_name, value in findings_met.items():
            metrics_data[f"findings_{metric_name}"] = value

        # 添加impression指标(如果是第二阶段)
        if train_stage != 1 and impression_met:
            for metric_name, value in impression_met.items():
                metrics_data[f"impression_{metric_name}"] = value

        # 保存评估指标，添加epoch信息
        metrics_df = pd.DataFrame([metrics_data])
        metrics_filename = f"{mode}_metrics_epoch_{epoch_str}.csv"
        metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
        logger.info(f"评估指标已保存到CSV文件: {metrics_filename}")

        # 返回结果
        result = {
            "findings_met": findings_met,
            "impression_met": impression_met,
            "loss": running_loss / len(data_loader),
            "results_df": results_df,
            "metrics_df": metrics_df,
        }

    return running_loss / len(data_loader), result


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
    # --- Model Parameters ---
    # model.load_state_dict(checkpoint["model_state_dict"])
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

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
    # if scheduler != None:
    #     try:
    #         scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     except:  # Input scheduler doesn't fit the checkpoint one --> should be ignored
    #         print("Cannot load the scheduler")
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
    train_stage=2,
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
        train_stage: 训练阶段，默认为2
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

            source["train_stage"] = train_stage
            source["mode"] = mode

            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])
            if train_stage != 1:
                impression_preds_list.extend([re for re in output["impression_text"]])

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

    # 添加impression相关数据(如果是第二阶段)
    if train_stage == 2:
        results_data["impression_gt"] = impression_gts_list
        results_data["impression_pred"] = impression_preds_list

    # 将结果转换为DataFrame并保存
    results_df = pd.DataFrame(results_data)

    # 生成文件名并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{mode}_generations_{timestamp}.csv"
    save_path = os.path.join(save_dir, csv_filename)
    results_df.to_csv(save_path, index=False)

    logger.info(f"生成结果已保存到: {save_path}")
    logger.info(f"总共保存了 {len(findings_gts_list)} 条记录")
