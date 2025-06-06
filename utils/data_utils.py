"""数据处理工具函数"""
import torch


def data_to_device(data, device="cpu"):
    """
    递归地将数据移动到指定设备
    
    Args:
        data: 数据（可以是tensor、list、tuple、dict）
        device: 目标设备
        
    Returns:
        移动到设备后的数据
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return tuple(data_to_device(item, device) for item in data)
    elif isinstance(data, list):
        return [data_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {k: data_to_device(v, device) for k, v in data.items()}
    return data


def data_concatenate(iterable_data, dim=0):
    """
    连接可迭代数据
    
    Args:
        iterable_data: 可迭代的数据
        dim: 连接维度
        
    Returns:
        连接后的数据
    """
    data = iterable_data[0]
    
    if isinstance(data, torch.Tensor):
        return torch.cat(list(iterable_data), dim=dim)
        
    elif isinstance(data, (tuple, list)):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        
        for col in range(num_cols):
            data_col = [iterable_data[row][col] for row in range(num_rows)]
            return_data.append(torch.cat(data_col, dim=dim))
            
        return tuple(return_data) if isinstance(data, tuple) else return_data
        
    elif isinstance(data, dict):
        return_data = {}
        for key in data.keys():
            data_col = [iterable_data[row][key] for row in range(len(iterable_data))]
            return_data[key] = torch.cat(data_col, dim=dim)
        return return_data
        
    else:
        raise TypeError("Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.")


def data_distributor(model, source):
    """
    根据source类型分发数据到模型
    
    Args:
        model: PyTorch模型
        source: 输入数据
        
    Returns:
        模型输出
    """
    if isinstance(source, torch.Tensor):
        return model(source)
    elif isinstance(source, (tuple, list)):
        return model(*source)
    elif isinstance(source, dict):
        return model(**source)
    else:
        raise TypeError("Unsupported DataType! Try List/Tuple/Dict!")


def args_to_kwargs(args, kwargs_list=None):
    """
    将参数转换为关键字参数
    
    Args:
        args: 参数
        kwargs_list: 关键字列表
        
    Returns:
        字典形式的参数
    """
    if kwargs_list is None:
        return args
        
    if isinstance(args, dict):
        return args
        
    # 确保args是列表
    if isinstance(args, torch.Tensor):
        args = [args]
        
    assert len(args) == len(kwargs_list), f"参数数量不匹配: {len(args)} vs {len(kwargs_list)}"
    return dict(zip(kwargs_list, args))


def prepare_batch_data(
    config,
    batch,
    data_loader,
    device,
    findings=True,
    history=True,
    label=True,
    bbox=True,
):
    """
    准备批次数据，对整个batch进行tokenization
    
    Args:
        config: 配置参数
        batch: 输入的批次数据
        data_loader: 数据加载器
        device: 计算设备
        findings: 是否处理findings字段
        history: 是否处理history字段
        label: 是否处理label字段
        bbox: 是否处理bbox_targets字段
        
    Returns:
        source_data, target_data, None
    """
    source = {}
    target = {}
    
    # 处理图像数据
    if "image" in batch:
        source["image"] = batch["image"].to(device, non_blocking=True)
    
    # 处理文本字段
    text_fields_to_process = []
    if findings and "findings" in batch:
        text_fields_to_process.append(("findings", config.MAX_LEN_FINDINGS))
    if history and "history" in batch:
        text_fields_to_process.append(("history", config.MAX_LEN_HISTORY))
    
    # 获取tokenizer
    tokenizer = data_loader.dataset.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'  # BERT使用右侧padding
    
    # 批量处理文本
    for field, max_len in text_fields_to_process:
        texts = batch[field]
        encoded = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device, non_blocking=True)
        
        source[field] = encoded
        target[field] = encoded
    
    # 恢复tokenizer设置
    tokenizer.padding_side = original_padding_side
    
    # 处理标签
    if label and "label" in batch:
        source["label"] = target["label"] = batch["label"].to(device, non_blocking=True)
    
    # 处理边界框
    if bbox and "bbox_targets" in batch:
        processed_bbox_targets = []
        for bbox_target in batch["bbox_targets"]:
            processed_target = {}
            for key, value in bbox_target.items():
                if isinstance(value, torch.Tensor):
                    processed_target[key] = value.to(device, non_blocking=True)
                else:
                    processed_target[key] = value
            processed_bbox_targets.append(processed_target)
        source["bbox_targets"] = processed_bbox_targets
    
    # 处理image_id
    if "image_id" in batch:
        source["image_ids"] = batch["image_id"]
    
    # 处理解剖区域嵌入
    if "anatomical_embeddings" in batch:
        source["anatomical_embeddings_batch"] = batch["anatomical_embeddings"]
    
    return source, target, None

