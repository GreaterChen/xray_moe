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
            # subset_size=10 if config.PHASE.startswith("TRAIN") else 100,
        )

        test_data = MIMIC(
            config.DATA_DIR,
            input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="test",
            # subset_size=100 if config.DEBUG else None,
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
        # 创建MOE模型
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
            _, _ = load(config.VIT_CHECKPOINT_PATH_FROM, vit_model, load_model="vit")

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
            # 注释掉冻结ViT的代码
            # for param in model.image_encoder.parameters():
            #     param.requires_grad = False

            # 计算每个模块的参数量
            module_parameters = {
                "Enhanced FastRCNN (frozen)": count_parameters(enhanced_rcnn),
                "ViT (trainable)": count_parameters(vit_model),
                "BERT Decoder": count_parameters(bert_model),
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
        # drop_last=True,
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

    # 在FINETUNE_MISTRAL或FINETUNE_LLAMA或FINETUNE_BERT阶段，只优化特定参数
    if config.PHASE.startswith("FINETUNE_"):
        # 优化解码器的参数和ViT的参数
        trainable_params = list(model.findings_decoder.parameters()) + list(model.image_encoder.parameters())
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
            config.CHECKPOINT_PATH_FROM, model, optimizer, scheduler, None
        )
        logger.info(
            f"Reloaded from {config.CHECKPOINT_PATH_FROM}: Last Epoch {last_epoch}, Best Metric {best_metric}, Test Metric {test_metric}"
        )

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
                images = batch["image"].cuda()
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
        process_dataset(val_loader, "val")
        
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
                device="cuda"
            )
            logger.info(f"CheXbert评估器初始化成功，使用检查点: {chexbert_path}")
        except Exception as e:
            logger.error(f"CheXbert评估器初始化失败: {e}")
            chexbert_metrics = None
        
        
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
                data_loader=test_loader,
                model=model,
                logger=logger,
                metric_ftns=compute_scores,
                mode="val",
                device="cuda",
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
            device="cuda",
            chexbert_metrics=chexbert_metrics,
        )

        # # 打印最终结果
        # logger.info(
        #     f"验证集 - BLEU-4: {final_val_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_val_result['report_generation_metrics']['rougeL']:.4f}"
        # )
        # logger.info(
        #     f"测试集 - BLEU-4: {final_test_result['report_generation_metrics']['bleu4']:.4f}, ROUGE-L: {final_test_result['report_generation_metrics']['rougeL']:.4f}"
        # )

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
