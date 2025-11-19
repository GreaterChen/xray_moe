"""
训练主脚本 - 重构版本
使用训练器模式，大幅简化代码结构
"""
import os
import warnings
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
from transformers import BertTokenizer
from transformers import logging as hf_logging

# 设置 tokenizers 环境变量，避免多进程时的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 屏蔽警告
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used")
# 屏蔽 torchvision 的 deprecation warnings
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than.*weights.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 避免DataLoader在多进程下创建过多mmap导致的内存映射失败
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# 项目模块
from utils import setup_logger
from device_utils import DeviceManager, setup_for_distributed
from datasets import MIMIC, mimic_collate_fn, IUXRAY, iuxray_collate_fn
from configs import config
from trainers import TrainerFactory


def resolve_local_hf_path(path, candidate_files=None, verbose=False):
    """
    解析本地HuggingFace缓存路径，必要时自动定位snapshots子目录
    """
    if not path:
        return path
    
    if isinstance(candidate_files, str):
        candidate_files = (candidate_files,)
    candidate_files = candidate_files or ("config.json",)
    
    def has_required_files(directory):
        for filename in candidate_files:
            if os.path.exists(os.path.join(directory, filename)):
                return True
        return False
    
    if os.path.isfile(path):
        return path
    
    if os.path.isdir(path):
        if has_required_files(path):
            return path
        
        snapshots_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshot_dirs = []
            for name in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, name)
                if os.path.isdir(candidate):
                    try:
                        mtime = os.path.getmtime(candidate)
                    except OSError:
                        mtime = 0
                    snapshot_dirs.append((mtime, candidate))
            for _, candidate in sorted(snapshot_dirs, key=lambda x: x[0], reverse=True):
                if has_required_files(candidate):
                    if verbose:
                        print(f"ℹ️ 检测到本地snapshot路径: {candidate}")
                    return candidate
        if verbose:
            print(f"⚠️ 未在路径 {path} 找到 {candidate_files}，将按原路径尝试加载。")
    
    return path


def setup_tokenizer(config):
    """
    根据配置创建tokenizer
    
    Args:
        config: 配置对象
        
    Returns:
        tokenizer对象
    """
    decoder_type = getattr(config, 'DECODER_TYPE', 'bert').lower()
    
    if decoder_type == 'qwenvl':
        # Qwen VL tokenizer
        from transformers import AutoTokenizer
        qwen_model_name = getattr(config, 'QWEN_MODEL_NAME', 'Qwen/Qwen3-VL-4B-Instruct')
        qwen_model_path = getattr(config, 'QWEN_MODEL_PATH', None)
        hf_cache_dir = getattr(config, 'HF_CACHE_DIR', None)
        local_files_only = getattr(config, 'HF_LOCAL_FILES_ONLY', False)
        
        tokenizer_source = qwen_model_path or qwen_model_name
        tokenizer_kwargs = {
            "trust_remote_code": True,
        }
        if hf_cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = hf_cache_dir
        
        tokenizer_candidate_files = (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "merges.txt",
        )
        resolved_tokenizer_path = resolve_local_hf_path(
            tokenizer_source,
            candidate_files=tokenizer_candidate_files,
            verbose=True,
        )
        
        if local_files_only or os.path.exists(resolved_tokenizer_path):
            tokenizer_kwargs["local_files_only"] = True
            local_files_only = True
        
        tokenizer_source = resolved_tokenizer_path
        
        load_from = "本地/缓存" if tokenizer_kwargs.get("local_files_only") else "远程"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **tokenizer_kwargs
        )
        # Qwen tokenizer已经有pad_token (<|endoftext|>)，不需要额外设置
        print(f"✅ 使用Qwen tokenizer: {tokenizer_source}")
        print(f"   加载来源: {load_from}")
        print(f"   pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    else:
        # BERT tokenizer (默认)
        from transformers import AutoTokenizer
        
        # 获取预训练模型名称（支持医学模型）
        bert_model = getattr(config, 'BERT_PRETRAINED_MODEL', 'bert-base-uncased')
        
        # 使用AutoTokenizer自动选择合适的tokenizer类
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                bert_model, 
                local_files_only=True
            )
            print(f"✅ 从本地加载tokenizer: {bert_model}")
        except:
            print(f"⚠️  本地没有tokenizer，从Huggingface下载: {bert_model}...")
            tokenizer = AutoTokenizer.from_pretrained(
                bert_model, 
                local_files_only=False
            )
            print(f"✅ Tokenizer下载完成: {bert_model}")
        
        # 添加特殊tokens：BOS用于解码开始，EOS用于生成结束
        special_tokens = {}
        if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
            special_tokens["bos_token"] = "[DEC]"
        if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
            special_tokens["eos_token"] = "[EOS]"
        
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
            
        # 设置left padding（用于decoder模型）
        tokenizer.padding_side = 'left'
        print(f"✅ 使用{bert_model.split('/')[-1]} tokenizer")
        print(f"   padding_side: {tokenizer.padding_side}")
        print(f"   pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   bos_token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"   eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    return tokenizer


def create_datasets(config, tokenizer):
    """
    创建数据集
    
    Args:
        config: 配置对象
        tokenizer: 分词器
        
    Returns:
        train_data, valid_data, test_data, dataset_type
    """
    input_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # 根据配置决定使用哪个数据集
    dataset_name = getattr(config, 'DATASET_NAME', 'MIMIC')
    
    if dataset_name == 'IUXRAY':
        # 使用IU_XRAY数据集
        # 加载共享数据
        IUXRAY.load_shared_data(
            ann_path=config.IUXRAY_ANN_PATH
        )
        
        # 创建训练数据集
        train_data = IUXRAY(
            ann_path=config.IUXRAY_ANN_PATH,
            images_dir=config.IUXRAY_IMAGES_DIR,
            input_size=input_size,
            random_transform=True,
            tokenizer=tokenizer,
            mode="train"
        )
        
        # 创建验证数据集
        valid_data = IUXRAY(
            ann_path=config.IUXRAY_ANN_PATH,
            images_dir=config.IUXRAY_IMAGES_DIR,
            input_size=input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="validate"
        )
        
        # 创建测试数据集
        test_data = IUXRAY(
            ann_path=config.IUXRAY_ANN_PATH,
            images_dir=config.IUXRAY_IMAGES_DIR,
            input_size=input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="test"
        )
        
        return train_data, valid_data, test_data, 'IUXRAY'
        
    else:
        # 使用MIMIC数据集（默认）
        # 加载共享数据
        MIMIC.load_shared_data(
            directory=config.DATA_DIR,
            ann_dir=config.ANN_DIR,
            mode=config.MODE,
            binary_mode=True,
            split_csv_path=config.SPLIT_CSV_PATH,
            generation_target=config.GENERATION_TARGET
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
            mode="train",
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
            mode="train",
            subset_size=3000,
            generation_target=config.GENERATION_TARGET
        )
        
        return train_data, valid_data, test_data, 'MIMIC'


def create_data_loaders(train_data, valid_data, test_data, config, device_manager, dataset_type='MIMIC'):
    """
    创建数据加载器
    
    Args:
        train_data: 训练数据集
        valid_data: 验证数据集
        test_data: 测试数据集
        config: 配置对象
        device_manager: 设备管理器
        dataset_type: 数据集类型 ('MIMIC' 或 'IUXRAY')
        
    Returns:
        train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler
    """
    # 根据数据集类型选择collate_fn
    collate_fn = iuxray_collate_fn if dataset_type == 'IUXRAY' else mimic_collate_fn
    
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
        pin_memory=getattr(config, 'PIN_MEMORY', True if device_manager.device.type == 'cuda' else False),
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 2),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', False) if config.NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )
    
    # 创建验证数据加载器
    valid_loader = data.DataLoader(
        valid_data,
        batch_size=config.VAL_BATCH_SIZE,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, 'PIN_MEMORY', True if device_manager.device.type == 'cuda' else False),
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 2),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', False) if config.NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )
    
    # 创建测试数据加载器
    test_loader = data.DataLoader(
        test_data,
        batch_size=config.VAL_BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, 'PIN_MEMORY', True if device_manager.device.type == 'cuda' else False),
        prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 2),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', False) if config.NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler


def main():
    """主函数"""
    try:
        # 1. 初始化设备管理器（必须最先执行，以便正确设置分布式环境）
        device_manager = DeviceManager(config)
        
        # 2. 设置分布式打印（只有主进程打印，避免重复输出）
        setup_for_distributed(device_manager.is_main_process())
        
        # 3. 初始化日志（只有主进程输出到控制台，所有信息都保存到文件）
        logger, log_file = setup_logger(
            log_dir="logs", 
            is_main_process=device_manager.is_main_process()
        )
        logger.info("=" * 80)
        logger.info("开始训练流程")
        logger.info("=" * 80)
        logger.info(f"日志保存位置: {log_file}")
        
        # 4. 打印设备信息
        logger.info("设备配置信息:")
        device_manager.print_info()
        
        # 5. 设置随机种子
        torch.manual_seed(config.SEED)
        if device_manager.distributed:
            torch.manual_seed(config.SEED + device_manager.rank)
        logger.info(f"随机种子已设置: {config.SEED}")
        
        # 6. 创建tokenizer
        logger.info("创建tokenizer...")
        tokenizer = setup_tokenizer(config)
        logger.info(f"Tokenizer词汇表大小: {len(tokenizer)}")
        
        # 7. 创建数据集
        logger.info("创建数据集...")
        train_data, valid_data, test_data, dataset_type = create_datasets(config, tokenizer)
        logger.info(f"数据集类型: {dataset_type}")
        logger.info(f"训练集大小: {len(train_data)}")
        logger.info(f"验证集大小: {len(valid_data)}")
        logger.info(f"测试集大小: {len(test_data)}")
        
        # 8. 创建数据加载器
        logger.info("创建数据加载器...")
        train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler = create_data_loaders(
            train_data, valid_data, test_data, config, device_manager, dataset_type
        )
        
        # 9. 使用工厂创建训练器
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
        
        # 10. 设置数据加载器
        trainer.train_loader = train_loader
        trainer.valid_loader = valid_loader
        trainer.test_loader = test_loader
        trainer.train_sampler = train_sampler
        trainer.valid_sampler = valid_sampler
        trainer.test_sampler = test_sampler
        
        # 11. 运行训练
        logger.info("开始训练...")
        trainer.run()
        
        logger.info("=" * 80)
        logger.info("训练流程完成!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        if 'logger' in locals():
            logger.warning("训练被用户中断 (Ctrl+C)")
        else:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger().warning("训练被用户中断 (Ctrl+C)")
        raise
    except Exception as e:
        if 'logger' in locals():
            logger.critical("训练过程发生致命错误:", exc_info=True)
            logger.critical(f"错误类型: {type(e).__name__}")
            logger.critical(f"错误信息: {str(e)}")
        else:
            import logging
            import traceback
            logging.basicConfig(level=logging.CRITICAL)
            logging.getLogger().critical(f"训练过程发生致命错误: {type(e).__name__}: {str(e)}")
            logging.getLogger().critical(traceback.format_exc())
        raise
    finally:
        # 确保清理资源
        if 'logger' in locals():
            logger.info("清理资源...")
        if 'device_manager' in locals() and device_manager.distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                if 'logger' in locals():
                    logger.info("分布式进程组已清理")


if __name__ == "__main__":
    main()

