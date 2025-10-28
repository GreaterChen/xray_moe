"""
训练主脚本 - 重构版本
使用训练器模式，大幅简化代码结构
"""
import os
import warnings
import torch
import torch.utils.data as data
from transformers import BertTokenizer
from transformers import logging as hf_logging

# 屏蔽警告
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used")

# 项目模块
from utils import setup_logger
from device_utils import DeviceManager, setup_for_distributed
from datasets import MIMIC, mimic_collate_fn
from configs import config
from trainers import TrainerFactory


def setup_tokenizer(config):
    """
    根据配置创建tokenizer
    
    Args:
        config: 配置对象
        
    Returns:
        tokenizer对象
    """
    # BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", 
        local_files_only=True
    )
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
    return tokenizer


def create_datasets(config, tokenizer):
    """
    创建数据集
    
    Args:
        config: 配置对象
        tokenizer: 分词器
        
    Returns:
        train_data, valid_data, test_data
    """
    input_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # 加载共享数据
    MIMIC.load_shared_data(
        directory=config.DATA_DIR,
        ann_dir=config.ANN_DIR,
        mode=config.MODE,
        binary_mode=True,
        split_csv_path=config.SPLIT_CSV_PATH
    )
    
    # 创建训练数据集
    train_data = MIMIC(
        directory=config.DATA_DIR,
        ann_dir=config.ANN_DIR,
        images_dir=config.IMAGES_DIR,
        input_size=input_size,
        random_transform=True,
        tokenizer=tokenizer,
        mode="train",
        subset_size=100 if config.DEBUG else None,
        generation_target=config.GENERATION_TARGET
    )
    
    # 创建验证数据集 
    valid_data = MIMIC(
        directory=config.DATA_DIR,
        ann_dir=config.ANN_DIR,
        images_dir=config.IMAGES_DIR,
        input_size=input_size,
        random_transform=False,
        tokenizer=tokenizer,
        mode="valid",
        subset_size=50 if config.DEBUG else None,
        generation_target=config.GENERATION_TARGET
    )
    
    # 创建测试数据集
    test_data = MIMIC(
        directory=config.DATA_DIR,
        ann_dir=config.ANN_DIR,
        images_dir=config.IMAGES_DIR,
        input_size=input_size,
        random_transform=False,
        tokenizer=tokenizer,
        mode="test",
        subset_size=50 if config.DEBUG else None,
        generation_target=config.GENERATION_TARGET
    )
    
    return train_data, valid_data, test_data


def create_data_loaders(train_data, valid_data, test_data, config, device_manager):
    """
    创建数据加载器
    
    Args:
        train_data: 训练数据集
        valid_data: 验证数据集
        test_data: 测试数据集
        config: 配置对象
        device_manager: 设备管理器
        
    Returns:
        train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler
    """
    # 获取分布式采样器
    train_sampler = device_manager.get_sampler(train_data, shuffle=True)
    valid_sampler = device_manager.get_sampler(valid_data, shuffle=False)
    test_sampler = device_manager.get_sampler(test_data, shuffle=False)
    
    # 创建训练数据加载器
    train_loader = data.DataLoader(
        train_data,
        batch_size=config.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device_manager.device.type == 'cuda' else False,
        collate_fn=mimic_collate_fn
    )
    
    # 创建验证数据加载器
    valid_loader = data.DataLoader(
        valid_data,
        batch_size=config.VAL_BATCH_SIZE,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device_manager.device.type == 'cuda' else False,
        collate_fn=mimic_collate_fn
    )
    
    # 创建测试数据加载器
    test_loader = data.DataLoader(
        test_data,
        batch_size=config.VAL_BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device_manager.device.type == 'cuda' else False,
        collate_fn=mimic_collate_fn
    )
    
    return train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler


def main():
    """主函数"""
    logger = setup_logger(log_dir="logs")
    logger.info("=" * 80)
    logger.info("开始训练流程")
    logger.info("=" * 80)
    
    # 1. 初始化设备管理器
    logger.info("初始化设备管理器...")
    device_manager = DeviceManager(config)
    device_manager.print_info()
    setup_for_distributed(device_manager.is_main_process())
    
    # 2. 设置随机种子
    torch.manual_seed(config.SEED)
    if device_manager.distributed:
        torch.manual_seed(config.SEED + device_manager.rank)
    logger.info(f"随机种子已设置: {config.SEED}")
    
    # 3. 创建tokenizer
    logger.info("创建tokenizer...")
    tokenizer = setup_tokenizer(config)
    logger.info(f"Tokenizer词汇表大小: {len(tokenizer)}")
    
    # 4. 创建数据集
    logger.info("创建数据集...")
    train_data, valid_data, test_data = create_datasets(config, tokenizer)
    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(valid_data)}")
    logger.info(f"测试集大小: {len(test_data)}")
    
    # 5. 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler = create_data_loaders(
        train_data, valid_data, test_data, config, device_manager
    )
    
    # 6. 使用工厂创建训练器
    logger.info(f"创建训练器 (阶段: {config.PHASE})...")
    try:
        trainer = TrainerFactory.create_trainer(
            config=config,
            device_manager=device_manager,
            logger=logger,
            tokenizer=tokenizer
        )
    except ValueError as e:
        logger.error(f"训练器创建失败: {e}")
        logger.info(f"支持的训练阶段: {TrainerFactory.list_supported_phases()}")
        return
    
    # 7. 设置数据加载器
    trainer.train_loader = train_loader
    trainer.valid_loader = valid_loader
    trainer.test_loader = test_loader
    trainer.train_sampler = train_sampler
    trainer.valid_sampler = valid_sampler
    trainer.test_sampler = test_sampler
    
    # 8. 运行训练
    logger.info("开始训练...")
    trainer.run()
    
    logger.info("=" * 80)
    logger.info("训练流程完成!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

