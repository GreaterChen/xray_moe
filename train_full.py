import json
import os
import numpy as np
import matplotlib

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

logger = setup_logger(log_dir="logs")

# --- Main Program ---
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    torch.manual_seed(config.SEED)

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

        MIMIC.load_shared_data(config.DATA_DIR, config.ANN_DIR, config.MODE)
        # 创建训练、验证和测试数据集
        train_data = MIMIC(
            config.DATA_DIR,
            input_size,
            random_transform=True,
            tokenizer=tokenizer,
            mode="train",
            subset_size=100 if config.DEBUG else None,
        )

        val_data = MIMIC(
            config.DATA_DIR,
            input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="val",
            subset_size=10 if config.PHASE.startswith("TRAIN") else 100,
        )

        test_data = MIMIC(
            config.DATA_DIR,
            input_size,
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

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN": count_parameters(enhanced_rcnn),
                "ViT": count_parameters(vit_model),
                "CXR BERT": count_parameters(cxr_bert),
            }
        elif config.PHASE == "INFER_BERT":
            cxr_bert = CXR_BERT_FeatureExtractor()
            model = MOE(config=config, cxr_bert=cxr_bert)

            # 计算每个模块的参数量
            module_parameters = {
                "CXR BERT": count_parameters(cxr_bert),
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
            for param in model.image_encoder.parameters():
                param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (frozen)": count_parameters(vit_model),
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
            for param in model.image_encoder.parameters():
                param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (frozen)": count_parameters(vit_model),
                "Llama 3.2 3B Generator": count_parameters(llama_model),
            }

        # 打印每个模块的参数量
        for module_name, param_count in module_parameters.items():
            logger.info(f"{module_name}: {param_count} parameters")

    else:
        raise ValueError("Invalid model_name")

    # Data loaders
    train_loader = data.DataLoader(
        train_data,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=mimic_collate_fn,
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=mimic_collate_fn,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=mimic_collate_fn,
    )

    # 打印数量
    logger.info(f"Train Data Size: {len(train_data)}")
    logger.info(f"Val Data Size: {len(val_data)}")
    logger.info(f"Test Data Size: {len(test_data)}")

    model = model.cuda()

    # 在FINETUNE_MISTRAL或FINETUNE_LLAMA阶段，只优化特定参数
    if config.PHASE == "FINETUNE_MISTRAL" or config.PHASE == "FINETUNE_LLAMA":
        # 只优化历史编码器和生成器的参数
        trainable_params = list(model.findings_decoder.parameters())
        optimizer = optim.AdamW(
            trainable_params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    scheduler = LinearWarmupCosineLRScheduler(
        optimizer,
        config.EPOCHS,
        config.MIN_LR,
        config.LEARNING_RATE,
        decay_rate=None,
        warmup_start_lr=config.WARMUP_LR,
        warmup_steps=config.WARMUP_STEPS,
    )

    logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    last_epoch = -1
    best_metric = -1e9

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Load checkpoint if needed
    if config.CHECKPOINT_PATH_FROM:
        last_epoch, (best_metric, test_metric) = load(
            config.CHECKPOINT_PATH_FROM, model, optimizer, scheduler
        )
        logger.info(
            f"Reloaded from {config.CHECKPOINT_PATH_FROM}: Last Epoch {last_epoch}, Best Metric {best_metric}, Test Metric {test_metric}"
        )

    metrics = compute_scores

    # Training phase
    if config.PHASE == "TRAIN_DETECTION" or config.PHASE == "PRETRAIN_VIT":
        if config.CHECKPOINT_PATH_FROM:
            _, _ = load(config.CHECKPOINT_PATH_FROM, model, optimizer, scheduler)

        criterion = None
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(last_epoch + 1, config.EPOCHS):
            print(f"Epoch: {epoch}")
            train_loss = train(
                config,
                train_loader,
                model,
                optimizer,
                criterion,
                config.EPOCHS,
                epoch,
                scheduler=scheduler,
                device="cuda",
                kw_src=config.KW_SRC,
                kw_tgt=config.KW_TGT,
                scaler=scaler,
                writer=writer,
            )
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
        for epoch in range(last_epoch + 1, config.EPOCHS):
            print(f"Epoch: {epoch}")
            train_loss = infer_bert(
                config,
                train_loader,
                model,
                num_epochs=config.EPOCHS,
                current_epoch=epoch,
                device="cuda",
                kw_src=config.KW_SRC,
                kw_tgt=config.KW_TGT,
            )

    elif config.PHASE == "FINETUNE_MISTRAL":
        # Mistral微调阶段
        if config.CHECKPOINT_PATH_FROM:
            _, _ = load(config.CHECKPOINT_PATH_FROM, model, optimizer, scheduler)
            logger.info(f"从 {config.CHECKPOINT_PATH_FROM} 加载模型权重")

        criterion = None
        scaler = torch.amp.GradScaler() if config.USE_MIXED_PRECISION else None

        for epoch in range(last_epoch + 1, config.EPOCHS):
            logger.info(f"Epoch: {epoch}")

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
                device="cuda",
                kw_src=config.KW_SRC,
                kw_tgt=config.KW_TGT,
                scaler=scaler,
                writer=writer,
            )

            # 测试
            test_loss, result = test_llm(
                config=config,
                data_loader=val_loader,
                model=model,
                logger=logger,
                metric_ftns=compute_scores,
                mode="val",
                device="cuda",
                epoch=epoch,
                writer=writer,
            )

            # 保存检查点
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

            # 每个epoch后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # 关闭TensorBoard writer
        writer.close()

        # 在验证集上进行最终评估
        logger.info("在验证集上进行最终评估...")
        final_val_loss, final_val_result = test_llm(
            config=config,
            data_loader=val_loader,
            model=model,
            logger=logger,
            metric_ftns=compute_scores,
            mode="val",
            device="cuda",
        )

        # 在测试集上进行最终评估
        logger.info("在测试集上进行最终评估...")
        final_test_loss, final_test_result = test_llm(
            config=config,
            data_loader=test_loader,
            model=model,
            logger=logger,
            metric_ftns=compute_scores,
            mode="test",
            device="cuda",
        )

        # 打印最终结果
        logger.info(
            f"验证集 - BLEU-4: {final_val_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_val_result['report_generation_metrics']['rougeL']:.4f}"
        )
        logger.info(
            f"测试集 - BLEU-4: {final_test_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_test_result['report_generation_metrics']['rougeL']:.4f}"
        )
        
    elif config.PHASE == "FINETUNE_LLAMA":
        # Llama微调阶段
        if config.CHECKPOINT_PATH_FROM:
            _, _ = load(config.CHECKPOINT_PATH_FROM, model.findings_decoder, optimizer, scheduler, load_model="decoder")
            logger.info(f"从 {config.CHECKPOINT_PATH_FROM} 加载模型权重")

        criterion = None
        scaler = torch.amp.GradScaler() if config.USE_MIXED_PRECISION else None

        for epoch in range(last_epoch + 1, config.EPOCHS):
            logger.info(f"Epoch: {epoch}")

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
                device="cuda",
                kw_src=config.KW_SRC,
                kw_tgt=config.KW_TGT,
                scaler=scaler,
                writer=writer,
            )

            # 测试
            test_loss, result = test_llm(
                config=config,
                data_loader=val_loader,
                model=model,
                logger=logger,
                metric_ftns=compute_scores,
                mode="val",
                device="cuda",
                epoch=epoch,
                writer=writer,
            )

            # 保存检查点
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

            # 每个epoch后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # 关闭TensorBoard writer
        writer.close()

        # 在验证集上进行最终评估
        logger.info("在验证集上进行最终评估...")
        final_val_loss, final_val_result = test_llm(
            config=config,
            data_loader=val_loader,
            model=model,
            logger=logger,
            metric_ftns=compute_scores,
            mode="val",
            device="cuda",
        )

        # 在测试集上进行最终评估
        logger.info("在测试集上进行最终评估...")
        final_test_loss, final_test_result = test_llm(
            config=config,
            data_loader=test_loader,
            model=model,
            logger=logger,
            metric_ftns=compute_scores,
            mode="test",
            device="cuda",
        )

        # 打印最终结果
        logger.info(
            f"验证集 - BLEU-4: {final_val_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_val_result['report_generation_metrics']['rougeL']:.4f}"
        )
        logger.info(
            f"测试集 - BLEU-4: {final_test_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_test_result['report_generation_metrics']['rougeL']:.4f}"
        )

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
            device="cuda",
            kw_src=config.KW_SRC,
            kw_tgt=config.KW_TGT,
        )

    else:
        raise ValueError("Invalid phase")
