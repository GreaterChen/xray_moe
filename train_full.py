import json
import os
import numpy as np
import matplotlib

# 添加以下代码来屏蔽特定警告
import warnings
from transformers import logging as hf_logging
# 屏蔽Hugging Face的warning
hf_logging.set_verbosity_error()  # 只显示错误，不显示警告
# 也可以使用Python的warnings模块
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used, but right-padding was detected")

matplotlib.use("Agg")  # 设置后端为Agg（非交互式）
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from datetime import datetime
import gc

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# --- Helper Packages ---
from tqdm import tqdm

# --- Project Packages ---
from utils import *
from device_utils import DeviceManager, to_device, setup_for_distributed
from datasets import MIMIC, mimic_collate_fn
from losses import *
from models.mrgn_model import *
from models.moe_model import *
from models.fast_rcnn_classifier import *
from models.vit import *
from models.mistral_finetuner import MistralFinetuner
from models.llama_finetuner import LlamaFinetuner
from metrics import compute_scores
from tools.optims import *
from configs import config
from tools.metrics_clinical import CheXbertMetrics

logger = setup_logger(log_dir="logs")

# --- Main Program ---
if __name__ == "__main__":
    # 初始化设备管理器
    device_manager = DeviceManager(config)
    device_manager.print_info()
    
    # 设置分布式训练的打印行为
    setup_for_distributed(device_manager.is_main_process())
    
    torch.manual_seed(config.SEED)
    if device_manager.distributed:
        torch.manual_seed(config.SEED + device_manager.rank)

    # Dataset-specific settings
    if config.DATASET_NAME == "MIMIC":
        input_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        if config.PHASE == "FINETUNE_MISTRAL":
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-v0.1", local_files_only=True  # 如果模型已下载到本地
            )
            tokenizer.pad_token = tokenizer.eos_token
        elif config.PHASE == "FINETUNE_LLAMA":
            tokenizer = AutoTokenizer.from_pretrained(
                "/home/chenlb/.cache/modelscope/hub/models/prithivMLmods/Llama-Doctor-3.2-3B-Instruct"
            )
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", local_files_only=True
            ) 
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        vocab_size = len(tokenizer)
        pad_id = tokenizer.pad_token_id

        MIMIC.load_shared_data(
            directory=config.DATA_DIR, 
            ann_dir=config.ANN_DIR, 
            mode=config.MODE, 
            extra_ann_dir=config.EXTRA_ANN_DIR,
            binary_mode=True
        )
        # 创建训练、验证和测试数据集
        train_data = MIMIC(
            directory=config.DATA_DIR,
            ann_dir=config.ANN_DIR,            
            input_size=input_size,
            random_transform=True,
            tokenizer=tokenizer,
            mode="train",
            subset_size=100 if config.DEBUG else None,
        )

        # val_data = MIMIC(
        #     directory=config.DATA_DIR,
        #     ann_dir=config.ANN_DIR,
        #     input_size=input_size,
        #     random_transform=False,
        #     tokenizer=tokenizer,
        #     mode="val",
        #     subset_size=10 if config.PHASE.startswith("TRAIN") else 100,
        # )

        test_data = MIMIC(
            directory=config.DATA_DIR,
            ann_dir=config.ANN_DIR,
            input_size=input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="test",
            subset_size=100 if config.DEBUG else None,
        )

        comment = f"Stage{config.PHASE}"
    else:
        raise ValueError("Invalid dataset_name")

    # Model-specific settings
    if config.MODEL_NAME == "MOE":
        # 初始化 TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = os.path.join(
            config.TENSORBOARD_DIR,
            f"{config.MODEL_NAME}_{config.PHASE}_{config.CHECKPOINT_PATH_TO.split('results/')[-1][:-1].replace('/', '_')}-{current_time}",
        )
        writer = SummaryWriter(tensorboard_log_dir)
        logger.info(f"TensorBoard 日志目录: {tensorboard_log_dir}")

        if config.PHASE == "TRAIN_DETECTION":
            fast_rcnn = DetectionOnlyFastRCNN()
            model = MOE(
                config=config,
                object_detector=fast_rcnn,
            )
            module_parameters = {
                "FastRCNN": count_parameters(fast_rcnn),
            }
        elif config.PHASE == "INFER_DETECTION":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(
                config.DETECTION_CHECKPOINT_PATH_FROM,
                detection_model,
                load_model="object_detector",
            )
            
            # 创建MOE模型，但只使用检测器部分
            model = MOE(
                config=config,
                object_detector=detection_model,
            )
            
            # 计算参数量
            module_parameters = {
                "FastRCNN (inference only)": count_parameters(detection_model),
            }
        elif config.PHASE == "PRETRAIN_VIT":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(config.DETECTION_CHECKPOINT_PATH_FROM, detection_model)

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            cxr_bert = CXR_BERT_FeatureExtractor()

            # 创建MOE模型
            model = MOE(
                config=config,
                object_detector=enhanced_rcnn,
                image_encoder=vit_model,
                cxr_bert=cxr_bert,
            )
            
            # 加载解剖区域数据库（用于区域级别对比学习）
            if getattr(config, 'ENABLE_REGION_ITC', True):
                anatomical_db_path = getattr(config, 'ANATOMICAL_DATABASE_PATH', None)
                if anatomical_db_path:
                    MIMIC.load_anatomical_embeddings(anatomical_db_path)
                    logger.info("✅ 解剖区域数据库已加载到MIMIC数据集中")
                else:
                    logger.warning("⚠️  未配置解剖区域数据库路径，区域级别ITC将被禁用")

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN": count_parameters(enhanced_rcnn),
                "ViT": count_parameters(vit_model),
                "CXR BERT": count_parameters(cxr_bert),
            }
        elif config.PHASE == "INFER_BERT":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            # 导入必要的模块
            from models.bert_cross_decoder import BertCrossDecoder
            from models.moe_bert_adapter import MoEBertAdapter

            # BERT解码器生成模型
            bert_model = MoEBertAdapter(
                config=config,
                tokenizer=tokenizer,
                hidden_dim=768,
                max_length=100
            )

            # 创建MOE模型（和FINETUNE_BERT相同的结构）
            model = MOE(
                config=config,
                object_detector=enhanced_rcnn,
                image_encoder=vit_model,
                findings_decoder=bert_model,
            )

            # 冻结所有参数（推理阶段不需要训练）
            for param in model.object_detector.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.findings_decoder.parameters():
                param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (frozen)": count_parameters(vit_model),
                "BERT Decoder (frozen)": count_parameters(bert_model),
            }
        elif config.PHASE == "FINETUNE_MISTRAL":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(
                config.DETECTION_CHECKPOINT_PATH_FROM,
                detection_model,
                load_model="object_detector",
            )

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            # 加载预训练的ViT模型
            _, _ = load(config.VIT_CHECKPOINT_PATH_FROM, vit_model, load_model="vit")

            # Mistral生成模型
            mistral_model = MistralFinetuner(
                config=config,
                base_model=config.MISTRAL_BASE_MODEL
                if hasattr(config, "MISTRAL_BASE_MODEL")
                else "mistralai/Mistral-7B-v0.1",
                lora_r=config.LORA_R if hasattr(config, "LORA_R") else 16,
                lora_alpha=config.LORA_ALPHA if hasattr(config, "LORA_ALPHA") else 32,
                lora_dropout=config.LORA_DROPOUT
                if hasattr(config, "LORA_DROPOUT")
                else 0.05,
                load_in_4bit=config.LOAD_IN_4BIT
                if hasattr(config, "LOAD_IN_4BIT")
                else True,
            )

            # 创建MOE模型
            model = MOE(
                config=config,
                object_detector=enhanced_rcnn,
                image_encoder=vit_model,
                findings_decoder=mistral_model,
            )

            # 冻结前两个阶段的模型参数
            for param in model.object_detector.parameters():
                param.requires_grad = False
            # 注释掉冻结ViT的代码
            # for param in model.image_encoder.parameters():
            #     param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (trainable)": count_parameters(vit_model),
                "Mistral Generator": count_parameters(mistral_model),
            }
            
        elif config.PHASE == "FINETUNE_LLAMA":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(
                config.DETECTION_CHECKPOINT_PATH_FROM,
                detection_model,
                load_model="object_detector",
            )

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            # 加载预训练的ViT模型
            _, _ = load(config.VIT_CHECKPOINT_PATH_FROM, vit_model, load_model="vit")

            # Llama 3.2 3B 生成模型
            llama_model = LlamaFinetuner(
                config=config,
                base_model="/home/chenlb/.cache/modelscope/hub/models/prithivMLmods/Llama-Doctor-3.2-3B-Instruct",
                lora_r=config.LORA_R if hasattr(config, "LORA_R") else 16,
                lora_alpha=config.LORA_ALPHA if hasattr(config, "LORA_ALPHA") else 32,
                lora_dropout=config.LORA_DROPOUT if hasattr(config, "LORA_DROPOUT") else 0.05,
                load_in_4bit=config.LOAD_IN_4BIT if hasattr(config, "LOAD_IN_4BIT") else True,
            )

            # 创建MOE模型
            model = MOE(
                config=config,
                object_detector=enhanced_rcnn,
                image_encoder=vit_model,
                findings_decoder=llama_model,
            )

            # 冻结前两个阶段的模型参数
            for param in model.object_detector.parameters():
                param.requires_grad = False
            # 注释掉冻结ViT的代码
            # for param in model.image_encoder.parameters():
            #     param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (trainable)": count_parameters(vit_model),
                "Llama 3.2 3B Generator": count_parameters(llama_model),
            }
        elif config.PHASE == "FINETUNE_BERT":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(
                config.DETECTION_CHECKPOINT_PATH_FROM,
                detection_model,
                load_model="object_detector",
            )

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            # 加载预训练的ViT模型
            # _, _ = load(config.VIT_CHECKPOINT_PATH_FROM, vit_model, load_model="vit")

            # 导入必要的模块
            from models.bert_cross_decoder import BertCrossDecoder
            from models.moe_bert_adapter import MoEBertAdapter

            # BERT解码器生成模型
            bert_model = MoEBertAdapter(
                config=config,
                tokenizer=tokenizer,
                hidden_dim=768,
                max_length=100
            )

            # 创建MOE模型
            model = MOE(
                config=config,
                object_detector=enhanced_rcnn,
                image_encoder=vit_model,
                findings_decoder=bert_model,
            )

            # 冻结前两个阶段的模型参数
            for param in model.object_detector.parameters():
                param.requires_grad = False
            
            # 冻结ViT的预训练参数，但保留LoRA参数可训练
            # for name, param in model.image_encoder.named_parameters():
            #     if 'lora_A' in name or 'lora_B' in name:
            #         # LoRA参数保持可训练
            #         param.requires_grad = True
            #     else:
            #         # 冻结其他所有参数（预训练的encoder、分类器等）
            #         param.requires_grad = False

            # 计算每个模块的参数量
            # 统计ViT中可训练和冻结的参数
            vit_trainable_params = sum(p.numel() for name, p in model.image_encoder.named_parameters() 
                                     if p.requires_grad and ('lora_A' in name or 'lora_B' in name))
            vit_frozen_params = sum(p.numel() for name, p in model.image_encoder.named_parameters() 
                                  if not p.requires_grad)
            
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT LoRA (trainable)": vit_trainable_params,
                "ViT Base (frozen)": vit_frozen_params,
                "BERT Decoder": count_parameters(bert_model),
            }

        elif config.PHASE == "BUILD_DATABASE":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(
                config.DETECTION_CHECKPOINT_PATH_FROM,
                detection_model,
                load_model="object_detector",
            )

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            # 加载预训练的ViT模型
            _, _ = load(config.VIT_CHECKPOINT_PATH_FROM, vit_model, load_model="vit")

            # 创建MOE模型（只需要检测器和ViT）
            model = MOE(
                config=config,
                object_detector=enhanced_rcnn,
                image_encoder=vit_model,
            )

            # 冻结所有参数，仅用于特征提取
            for param in model.parameters():
                param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (frozen)": count_parameters(vit_model),
            }

        # 打印每个模块的参数量
        for module_name, param_count in module_parameters.items():
            logger.info(f"{module_name}: {param_count} parameters")
    else:
        raise ValueError("Invalid model_name")

    # Data loaders
    # 获取分布式采样器
    train_sampler = device_manager.get_sampler(train_data, shuffle=True)
    
    train_loader = data.DataLoader(
        train_data,
        batch_size=config.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # 如果有sampler就不能设置shuffle
        num_workers=config.NUM_WORKERS,
        # prefetch_factor=2,
        pin_memory=True if device_manager.device.type == 'cuda' else False,
        # drop_last=True,
        collate_fn=mimic_collate_fn,
    )
    # val_loader = data.DataLoader(
    #     val_data,
    #     batch_size=config.VAL_BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=config.NUM_WORKERS,
    #     collate_fn=mimic_collate_fn,
    # )
    # 测试集采样器
    test_sampler = device_manager.get_sampler(test_data, shuffle=False)
    
    test_loader = data.DataLoader(
        test_data,
        batch_size=config.VAL_BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device_manager.device.type == 'cuda' else False,
        collate_fn=mimic_collate_fn,
    )

    # 打印数量
    logger.info(f"Train Data Size: {len(train_data)}")
    # logger.info(f"Val Data Size: {len(val_data)}")
    logger.info(f"Test Data Size: {len(test_data)}")

    # 使用设备管理器包装模型（自动处理单卡/多卡）
    model = device_manager.wrap_model(model)

    # 在FINETUNE_MISTRAL或FINETUNE_LLAMA或FINETUNE_BERT阶段，只优化特定参数
    if config.PHASE.startswith("FINETUNE_"):
        # 优化解码器的参数和ViT中可训练的参数（主要是LoRA参数）
        trainable_params = []
        # 添加解码器的所有参数
        trainable_params.extend([p for p in model.findings_decoder.parameters() if p.requires_grad])
        # 添加ViT中可训练的参数（在FINETUNE_BERT阶段主要是LoRA参数）
        trainable_params.extend([p for p in model.image_encoder.parameters() if p.requires_grad])
        
        logger.info(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
        
        # 根据GPU数量调整学习率
        adjusted_lr = device_manager.adjust_learning_rate(config.LEARNING_RATE) if config.ADJUST_LR_FOR_MULTI_GPU else config.LEARNING_RATE
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=adjusted_lr,
            weight_decay=config.WEIGHT_DECAY,
        )
        
        if device_manager.is_main_process():
            logger.info(f"原始学习率: {config.LEARNING_RATE}, 调整后学习率: {adjusted_lr}")
    elif config.PHASE in ['BUILD_DATABASE', 'INFER_BERT']:
        optimizer = None
    else:
        # 根据GPU数量调整学习率
        adjusted_lr = device_manager.adjust_learning_rate(config.LEARNING_RATE) if config.ADJUST_LR_FOR_MULTI_GPU else config.LEARNING_RATE
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=adjusted_lr,
            weight_decay=config.WEIGHT_DECAY,
        )
        
        if device_manager.is_main_process():
            logger.info(f"原始学习率: {config.LEARNING_RATE}, 调整后学习率: {adjusted_lr}")

    if optimizer is not None:
        scheduler = LinearWarmupCosineLRScheduler(
            optimizer,
            config.EPOCHS,
            config.MIN_LR,
            config.LEARNING_RATE,
            decay_rate=None,
            warmup_start_lr=config.WARMUP_LR,
            warmup_steps=config.WARMUP_STEPS,
        )
    else:
        scheduler = None

    logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    last_epoch = -1
    best_metric = -1e9

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Load checkpoint if needed (但在INFER_BERT阶段将在后面单独处理)
    if config.CHECKPOINT_PATH_FROM and config.PHASE != "INFER_BERT":
        loaded_data = load(
            config.CHECKPOINT_PATH_FROM, model, optimizer, scheduler, load_model="full"
        )
        if isinstance(loaded_data, tuple):
            last_epoch, metrics = loaded_data
            if metrics is not None and isinstance(metrics, tuple):
                best_metric, test_metric = metrics
                logger.info(
                    f"从 {config.CHECKPOINT_PATH_FROM} 加载模型权重: 上次训练轮次 {last_epoch}, 最佳指标 {best_metric}, 测试指标 {test_metric}"
                )
            else:
                logger.info(
                    f"从 {config.CHECKPOINT_PATH_FROM} 加载模型权重: 上次训练轮次 {last_epoch}, 无指标数据"
                )
        else:
            last_epoch = -1
            logger.info(f"从 {config.CHECKPOINT_PATH_FROM} 加载模型权重失败，将从头开始训练")
    metrics = compute_scores

    # INFER_DETECTION阶段：用于生成所有训练和验证集的bbox并保存为json文件
    if config.PHASE == "INFER_DETECTION":
        logger.info("开始INFER_DETECTION阶段：推断所有样本的bbox...")
        
        # 确保模型处于评估模式
        model.eval()
        
        # 创建存储检测结果的字典
        detection_results = {}
        
        # 定义处理单个数据集的函数
        def process_dataset(data_loader, split_name):
            logger.info(f"处理{split_name}数据集...")
            
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Processing {split_name}")):
                images = to_device(batch["image"], device_manager.device)
                image_paths = batch["image_path"]
                
                # 执行推断，获取检测结果
                with torch.no_grad():
                    if hasattr(model.object_detector, "predict_regions"):
                        # 如果是DetectionOnlyFastRCNN类实例
                        detections = model.object_detector.predict_regions(images)
                    else:
                        # 如果目标检测器在MOE模型内部
                        outputs = model.object_detector(images)
                        detections = outputs
                
                # 对每个样本进行处理
                for i, (img_path, detection) in enumerate(zip(image_paths, detections)):
                    # 提取图像ID
                    image_id = os.path.basename(img_path).split('.')[0]
                    
                    # 将检测结果转换为numpy数组，方便保存为json
                    boxes = detection["boxes"].cpu().numpy().tolist()
                    labels = detection["labels"].cpu().numpy().tolist()
                    scores = detection["scores"].cpu().numpy().tolist()
                    
                    # 保存到字典中
                    detection_results[image_id] = {
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores
                    }
        
        # 处理训练集
        process_dataset(train_loader, "train")
        
        # 处理验证集
        # process_dataset(val_loader, "val")
        
        # 处理测试集
        process_dataset(test_loader, "test")
        
        # 保存结果到json文件
        output_path = os.path.join(config.CHECKPOINT_PATH_TO, "detection_results.json")
        with open(output_path, 'w') as f:
            json.dump(detection_results, f)
        
        logger.info(f"检测结果已保存到 {output_path}")
        logger.info("INFER_DETECTION阶段完成")

    # Training phase
    elif config.PHASE == "TRAIN_DETECTION" or config.PHASE == "PRETRAIN_VIT":
        criterion = None
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(last_epoch + 1, config.EPOCHS):
            print(f"Epoch: {epoch}")
            
            # 分布式训练时设置epoch到sampler
            if device_manager.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            _, _ = load(
                config.DETECTION_CHECKPOINT_PATH_FROM,
                model,
                load_model=None,
            )
                
            # train_loss = train(
            #     config,
            #     train_loader,
            #     model,
            #     optimizer,
            #     criterion,
            #     config.EPOCHS,
            #     epoch,
            #     scheduler=scheduler,
            #     device=device_manager.device,
            #     kw_src=config.KW_SRC,
            #     kw_tgt=config.KW_TGT,
            #     scaler=scaler,
            #     writer=writer,
            #     device_manager=device_manager,  # 传递设备管理器
            # )
            if config.PHASE == "TRAIN_DETECTION":
                test_loss, result = test_detection(
                    config=config,
                    model=model.object_detector,
                    data_loader=test_loader,
                    logger=logger,
                    epoch=epoch,
                    writer=writer,
                )
                save_path = os.path.join(
                    config.CHECKPOINT_PATH_TO,
                    f'epoch_{epoch}_{result["overall_metrics"]["mAP"]}.pth',
                )

            elif config.PHASE == "PRETRAIN_VIT":
                test_loss, result = test_vit(
                    config=config,
                    model=model,
                    data_loader=test_loader,
                    logger=logger,
                    mode="test",
                    epoch=epoch,
                    writer=writer,
                )
                save_path = os.path.join(
                    config.CHECKPOINT_PATH_TO,
                    f'epoch_{epoch}_image_acc_{result["overall_metrics"]["ce_f1"]:.4f}.pth',
                )
            else:
                pass

            save(
                save_path,
                model,
                optimizer,
                scheduler,
                epoch,
                (test_loss, result),
            )

        # 关闭 TensorBoard writer
        writer.close()
    elif config.PHASE == "INFER_BERT":
        # 确保提供了checkpoint路径
        if not config.CHECKPOINT_PATH_FROM:
            raise ValueError("INFER_BERT阶段必须提供checkpoint路径用于推理!")

        # 🔧 修复权重加载问题：使用与FINETUNE_BERT相同的加载方式
        # 分别加载不同组件的权重，而不是一次性加载整个模型
        
        # 1. 加载图像编码器权重（ViT）
        if hasattr(config, 'IMAGE_ENCODER_CHECKPOINT_PATH_FROM') and config.IMAGE_ENCODER_CHECKPOINT_PATH_FROM:
            _, _ = load(config.IMAGE_ENCODER_CHECKPOINT_PATH_FROM, model.image_encoder, load_model="vit")
            logger.info(f"从 {config.IMAGE_ENCODER_CHECKPOINT_PATH_FROM} 加载图像编码器权重")
        
        # 2. 加载目标检测器权重  
        if hasattr(config, 'DETECTION_CHECKPOINT_PATH_FROM') and config.DETECTION_CHECKPOINT_PATH_FROM:
            _, _ = load(config.DETECTION_CHECKPOINT_PATH_FROM, model.object_detector, load_model="object_detector")
            logger.info(f"从 {config.DETECTION_CHECKPOINT_PATH_FROM} 加载目标检测器权重")
        
        # 3. 🚀 关键修复：使用与FINETUNE_BERT相同的方式加载BERT解码器权重
        logger.info("正在加载BERT解码器权重...")
        _, _ = load(config.CHECKPOINT_PATH_FROM, model.findings_decoder.decoder, load_model="decoder")
        logger.info(f"从 {config.CHECKPOINT_PATH_FROM} 加载BERT解码器权重到 model.findings_decoder.decoder")
        
        # 4. 如果有其他组件需要加载，可以在这里添加
        
        logger.info("权重加载完成！")
        
        # 🔧 确保模型处于评估模式
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        logger.info("模型已设置为评估模式，所有参数已冻结")

        # 检查是否存在缓存的检测结果
        detection_cache_path = os.path.join(config.CHECKPOINT_PATH_TO, "detection_cache.json")
        use_cached_detections = False
        
        if os.path.exists(detection_cache_path):
            logger.info(f"发现检测结果缓存文件: {detection_cache_path}")
            try:
                with open(detection_cache_path, 'r') as f:
                    detection_cache = json.load(f)
                use_cached_detections = True
                logger.info(f"成功加载 {len(detection_cache)} 个样本的检测缓存")
            except Exception as e:
                logger.warning(f"加载检测缓存失败: {e}，将重新进行检测")
                use_cached_detections = False
        else:
            logger.info("未找到检测结果缓存，将进行首次检测并保存缓存")
            use_cached_detections = False
        
        # 如果没有缓存，先进行bbox检测并保存（只用目标检测器）
        if not use_cached_detections:
            logger.info("正在生成bbox检测缓存...")
            detection_cache = {}
            
            # 创建独立的检测器用于bbox预测
            cache_detector = DetectionOnlyFastRCNN()
            _, _ = load(config.DETECTION_CHECKPOINT_PATH_FROM, cache_detector, load_model="object_detector")
            cache_detector = cache_detector.to(device_manager.device)
            cache_detector.eval()
            
            with torch.no_grad():
                cache_progress = tqdm(test_loader, desc="生成bbox缓存")
                for batch_idx, batch in enumerate(cache_progress):
                    images = to_device(batch["image"], device_manager.device)
                    image_paths = batch["image_path"]
                    
                    # 只进行目标检测，获取bbox预测
                    detections = cache_detector.predict_regions(images)
                    
                    # 保存每个样本的bbox预测结果
                    for i, img_path in enumerate(image_paths):
                        image_id = os.path.basename(img_path).split('.')[0]
                        
                        # 提取单个样本的bbox预测（29个区域）
                        sample_detection = {
                            "boxes": detections[i]["boxes"].cpu().numpy().tolist(),  # 预测的bbox坐标
                            "labels": detections[i]["labels"].cpu().numpy().tolist(),  # 区域标签
                            "scores": detections[i]["scores"].cpu().numpy().tolist()   # 置信度分数
                        }
                        
                        detection_cache[image_id] = sample_detection
            
            # 清理临时检测器
            del cache_detector
            torch.cuda.empty_cache()
            
            # 保存缓存到文件
            os.makedirs(os.path.dirname(detection_cache_path), exist_ok=True)
            with open(detection_cache_path, 'w') as f:
                json.dump(detection_cache, f)
            logger.info(f"bbox检测缓存已保存到: {detection_cache_path}")

        # 确保所有参数都被冻结（推理模式）
        for param in model.parameters():
            param.requires_grad = False

        # 初始化CheXbert评估器（如果需要）
        chexbert_metrics = None
        if hasattr(config, 'CHEXBERT_CHECKPOINT_PATH') and config.CHEXBERT_CHECKPOINT_PATH:
            try:
                from tools.metrics_clinical import CheXbertMetrics
                chexbert_metrics = CheXbertMetrics(
                    checkpoint_path=config.CHEXBERT_CHECKPOINT_PATH,
                    mbatch_size=config.VAL_BATCH_SIZE,
                    device=str(device_manager.device)
                )
                logger.info("CheXbert评估器初始化成功")
            except Exception as e:
                logger.warning(f"CheXbert评估器初始化失败: {e}")

        # 设置使用缓存的标志，传递给模型
        model.use_detection_cache = True
        model.detection_cache = detection_cache
        logger.info("启用bbox缓存模式进行推理...")

        # 在测试集上进行推理和评估
        logger.info("开始INFER_BERT阶段：在测试集上进行推理和评估...")
        
        test_loss, result = test_llm(
            config=config,
            data_loader=test_loader,
            model=model,
            logger=logger,
            metric_ftns=compute_scores,
            mode="test",
            device=device_manager.device,
            chexbert_metrics=chexbert_metrics,
        )

        # 打印推理结果
        if result:
            logger.info("=== INFER_BERT 推理结果 ===")
            if "report_generation_metrics" in result:
                metrics = result["report_generation_metrics"]
                logger.info(f"BLEU-1: {metrics.get('BLEU_1', 'N/A'):.4f}")
                logger.info(f"BLEU-4: {metrics.get('BLEU_4', 'N/A'):.4f}")
                logger.info(f"ROUGE-L: {metrics.get('ROUGE_L', 'N/A'):.4f}")
                
            if "chexbert_metrics" in result:
                chexbert = result["chexbert_metrics"]
                logger.info(f"CheXbert CE F1: {chexbert.get('ce_f1', 'N/A'):.4f}")
                logger.info(f"CheXbert CE Precision: {chexbert.get('ce_precision', 'N/A'):.4f}")
                logger.info(f"CheXbert CE Recall: {chexbert.get('ce_recall', 'N/A'):.4f}")

        # 保存生成结果
        logger.info("保存推理生成结果...")
        save_generations(
            config,
            test_loader,
            model,
            logger,
            save_dir=os.path.join(config.CHECKPOINT_PATH_TO, "infer_bert_generations"),
            mode="test",
            device=device_manager.device,
            kw_src=config.KW_SRC,
            kw_tgt=config.KW_TGT,
        )

    # 统一处理所有微调阶段(MISTRAL/LLAMA/BERT)
    elif config.PHASE.startswith("FINETUNE_"):
        # 微调阶段
        if config.DECODER_CHECKPOINT_PATH_FROM:
            if config.PHASE == "FINETUNE_LLAMA":
                # LLAMA使用特殊的加载器
                _, _ = load(config.DECODER_CHECKPOINT_PATH_FROM, model.findings_decoder, optimizer, scheduler, load_model="decoder")
            elif config.PHASE == "FINETUNE_BERT":
                # BERT使用标准加载器，但可以指定只加载decoder部分
                _, _ = load(config.DECODER_CHECKPOINT_PATH_FROM, model.findings_decoder.decoder, optimizer, scheduler, load_model="decoder")
            else:
                # MISTRAL使用标准加载器
                _, _ = load(config.DECODER_CHECKPOINT_PATH_FROM, model, optimizer, scheduler)
            logger.info(f"从 {config.DECODER_CHECKPOINT_PATH_FROM} 加载模型权重")

        # 初始化CheXbert评估器
        logger.info("初始化CheXbert评估器...")
        chexbert_path = config.CHEXBERT_CHECKPOINT_PATH
        try:
            chexbert_metrics = CheXbertMetrics(
                checkpoint_path=chexbert_path,
                mbatch_size=config.VAL_BATCH_SIZE,
                device=str(device_manager.device)
            )
            logger.info(f"CheXbert评估器初始化成功，使用检查点: {chexbert_path}")
        except Exception as e:
            logger.error(f"CheXbert评估器初始化失败: {e}")
            chexbert_metrics = None
        
        
        criterion = None
        scaler = torch.amp.GradScaler() if config.USE_MIXED_PRECISION else None

        for epoch in range(last_epoch + 1, config.EPOCHS):
            logger.info(f"Epoch: {epoch}")
            
            # 分布式训练时设置epoch到sampler
            if device_manager.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # 训练
            train_loss = train(
                config,
                train_loader,
                model,
                optimizer,
                criterion,
                config.EPOCHS,
                epoch,
                scheduler=scheduler,
                device=device_manager.device,
                kw_src=config.KW_SRC,
                kw_tgt=config.KW_TGT,
                scaler=scaler,
                writer=writer,
                device_manager=device_manager,  # 传递设备管理器
            )

            # 每5轮进行一次测试
            if (epoch + 1) % 5 == 0 or epoch == config.EPOCHS - 1:
            # if epoch == config.EPOCHS - 1:
                # 测试
                test_loss, result = test_llm(
                    config=config,
                    data_loader=test_loader,
                    model=model,
                    logger=logger,
                    metric_ftns=compute_scores,
                    mode="test",
                    device=device_manager.device,
                    epoch=epoch,
                    writer=writer,
                    chexbert_metrics=chexbert_metrics,
                )

                # 保存检查点 - 使用CheXbert指标如果可用
                if chexbert_metrics is not None and "chexbert_metrics" in result and "ce_f1" in result["chexbert_metrics"]:
                    save_path = os.path.join(
                        config.CHECKPOINT_PATH_TO,
                        f'epoch_{epoch}_bleu_{result["report_generation_metrics"]["BLEU_1"]:.4f}_ce_f1_{result["chexbert_metrics"]["ce_f1"]:.4f}.pth',
                    )
                else:
                    save_path = os.path.join(
                        config.CHECKPOINT_PATH_TO,
                        f'epoch_{epoch}_bleu_{result["report_generation_metrics"]["BLEU_1"]:.4f}.pth',
                    )

                save(
                    save_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    (test_loss, result),
                )
            else:
                # 非测试轮次也保存检查点，但不包含测试指标
                save_path = os.path.join(
                    config.CHECKPOINT_PATH_TO,
                    f'epoch_{epoch}.pth',
                )
                save(
                    save_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    None,  # 不包含测试结果
                )

            # 每个epoch后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # 关闭TensorBoard writer
        writer.close()

        # 在测试集上进行最终评估
        logger.info("在测试集上进行最终评估...")
        final_test_loss, final_test_result = test_llm(
            config=config,
            data_loader=test_loader,
            model=model,
            logger=logger,
            metric_ftns=compute_scores,
            mode="test",
            device=device_manager.device,
            chexbert_metrics=chexbert_metrics,
        )

        # # 打印最终结果
        # logger.info(
        #     f"验证集 - BLEU-4: {final_val_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_val_result['report_generation_metrics']['rougeL']:.4f}"
        # )
        # logger.info(
        #     f"测试集 - BLEU-4: {final_test_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_test_result['report_generation_metrics']['rougeL']:.4f}"
        # )

    elif config.PHASE == "BUILD_DATABASE":
        # BUILD_DATABASE阶段：构建解剖区域特征数据库
        logger.info("开始BUILD_DATABASE阶段：构建解剖区域特征数据库...")
        
        from utils import build_anatomical_database
        
        # 构建数据库
        build_anatomical_database(
            config=config,
            model=model,
            data_loader=train_loader,
            logger=logger,
            device=device_manager.device,
        )
        
        logger.info("解剖区域特征数据库构建完成！")

    elif config.MODE == "TEST":
        # 确保提供了checkpoint路径
        if not config.CHECKPOINT_PATH_FROM:
            raise ValueError("必须提供checkpoint路径用于测试!")

        # 加载模型权重
        _, _ = load(config.CHECKPOINT_PATH_FROM, model, optimizer, scheduler)
        logger.info(f"从 {config.CHECKPOINT_PATH_FROM} 加载模型权重")

        # 保存生成结果
        save_generations(
            config,
            test_loader,
            model,
            logger,
            save_dir=os.path.join(config.CHECKPOINT_PATH_TO, "generations"),
            mode="test",
            device=device_manager.device,
            kw_src=config.KW_SRC,
            kw_tgt=config.KW_TGT,
        )

    else:
        raise ValueError("Invalid phase")
