import json
import os
import numpy as np
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
import argparse

# --- Project Packages ---
from utils import *
from datasets import MIMIC, mimic_collate_fn
from losses import *
from models.mrgn_model import *
from models.moe_model import *
from models.fast_rcnn_classifier import *
from models.vit import *
from metrics import compute_scores
from tools.optims import *

logger = setup_logger(log_dir="logs")


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False, help="Debug mode.")

    # Data input settings
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/chenlb/MOE/",
        help="Root directory.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/chenlb/datasets/mimic_cxr/",
        help="Path to the directory.",
    )
    parser.add_argument(
        "--ann_dir",
        type=str,
        default="/mnt/chenlb/datasets/mimic_cxr/mimic_annotation_moe_bbox_filtered_split_numeric.json",
        help="Path to the annotation file.",
    )

    parser.add_argument(
        "--negative_pool_dir",
        type=str,
        default="/home/chenlb/MOE/results/ltc/negative_pool/pool.npy",
        help="Path to load the negative pool.",
    )

    parser.add_argument(
        "--model_name", type=str, default="MOE", choices=["MOE"], help="模型名称"
    )

    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MIMIC",
        choices=["MIMIC"],
        help="Dataset name to use.",
    )
    parser.add_argument(
        "--max_len_findings",
        type=int,
        default=100,
        help="Maximum length of the input text.",
    )

    parser.add_argument(
        "--max_len_history",
        type=int,
        default=50,
        help="Maximum length of the input text.",
    )

    parser.add_argument(
        "--tokenizer_max_len",
        type=int,
        default=30523,
        help="Maximum length of the tokenizer.",
    )

    parser.add_argument(
        "--kw_src",
        type=str,
        nargs="+",
        default=["image", "findings", "history", "bbox_targets"],
        help="Keyword arguments for the source inputs of the model (e.g., image, findings, impression).",
    )
    parser.add_argument(
        "--kw_tgt",
        type=str,
        nargs="+",
        default=["findings", "label"],
        help="Keyword arguments for the target outputs of the model (e.g., findings, impression).",
    )

    # Training settings
    parser.add_argument(
        "--phase",
        type=str,
        default="PRETRAIN_VIT",
        choices=["TRAIN_DETECTION", "INFER_BERT", "PRE_TRAIN_VIT", "TEST", "INFER"],
        help="Phase of the program",
    )

    # TRAIN OR TEST
    parser.add_argument(
        "--mode",
        type=str,
        default="TRAIN",
        choices=["TRAIN", "TEST"],
        help="Train or Test",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Batch size for validation."
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of workers for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate."
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=5e-6, help="Warmup learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps.")
    parser.add_argument(
        "--wd", type=float, default=0.01, help="Weight decay (L2 regularization)."
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # Device settings
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0", help="CUDA visible devices."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--detection_checkpoint_path_from",
        type=str,
        default="/home/chenlb/MOE/results/detection/epoch_9_BLEU_1_0.8791605068503925.pth",
        help="Path to load the detection checkpoint from.",
    )

    parser.add_argument(
        "--checkpoint_path_from",
        type=str,
        default=None,
        help="Path to load the checkpoint from.",
    )
    parser.add_argument(
        "--checkpoint_path_to",
        type=str,
        default="/home/chenlb/MOE/results/ltc/region_disease_support/",
        help="Path to save the checkpoint to.",
    )

    # TensorBoard settings
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="TensorBoard 日志目录",
    )

    args = parser.parse_args()

    # Convert args to dictionary
    args_dict = vars(args)

    # Create save path
    from pathlib import Path

    save_dir = Path(args.checkpoint_path_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "args_config.json"

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    return args


# --- Main Program ---
if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    torch.manual_seed(args.seed)

    # Dataset-specific settings
    if args.dataset_name == "MIMIC":
        input_size = (args.image_size, args.image_size)
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        vocab_size = len(tokenizer)
        pad_id = tokenizer.pad_token_id

        MIMIC.load_shared_data(args.data_dir, args.ann_dir, args.mode)
        # 创建训练、验证和测试数据集
        train_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=True,
            tokenizer=tokenizer,
            mode="train",
            subset_size=1000 if args.debug else None,
        )

        val_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="val",
            subset_size=10 if args.phase.startswith("TRAIN") else 100,
        )

        test_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=False,
            tokenizer=tokenizer,
            mode="test",
            subset_size=100 if args.debug else None,
        )

        comment = f"Stage{args.phase}"
    else:
        raise ValueError("Invalid dataset_name")

    # Model-specific settings
    if args.model_name == "MOE":

        # 初始化 TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = os.path.join(
            args.tensorboard_dir,
            f"{args.model_name}_{args.phase}_{args.checkpoint_path_to.split('results/')[-1][:-1].replace('/', '_')}-{current_time}",
        )
        writer = SummaryWriter(tensorboard_log_dir)
        logger.info(f"TensorBoard 日志目录: {tensorboard_log_dir}")

        if args.phase == "TRAIN_DETECTION":
            fast_rcnn = DetectionOnlyFastRCNN()
            model = MOE(
                args=args,
                object_detector=fast_rcnn,
            )
            module_parameters = {
                "FastRCNN": count_parameters(fast_rcnn),
            }
        elif args.phase == "PRETRAIN_VIT":
            # 初始化检测器
            detection_model = DetectionOnlyFastRCNN()
            _, _ = load(args.detection_checkpoint_path_from, detection_model)

            # 创建增强型FastRCNN
            enhanced_rcnn = EnhancedFastRCNN(
                pretrained_detector=detection_model, num_regions=29, feature_dim=768
            )

            # 初始化ViT模型
            vit_model = MedicalVisionTransformer()

            cxr_bert = CXR_BERT_FeatureExtractor()

            # 创建MOE模型
            model = MOE(
                args=args,
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
        elif args.phase == "INFER_BERT":
            cxr_bert = CXR_BERT_FeatureExtractor()
            model = MOE(args=args, cxr_bert=cxr_bert)

            # 计算每个模块的参数量
            module_parameters = {
                "CXR BERT": count_parameters(cxr_bert),
            }

        # 打印每个模块的参数量
        for module_name, param_count in module_parameters.items():
            logger.info(f"{module_name}: {param_count} parameters")

    else:
        raise ValueError("Invalid model_name")

    # Data loaders
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=mimic_collate_fn,
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mimic_collate_fn,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mimic_collate_fn,
    )

    # 打印数量
    logger.info(f"Train Data Size: {len(train_data)}")
    logger.info(f"Val Data Size: {len(val_data)}")
    logger.info(f"Test Data Size: {len(test_data)}")

    model = model.cuda()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer,
        args.epochs,
        args.min_lr,
        args.lr,
        decay_rate=None,
        warmup_start_lr=args.warmup_lr,
        warmup_steps=args.warmup_steps,
    )

    logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    last_epoch = -1
    best_metric = -1e9

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Load checkpoint if needed
    if args.checkpoint_path_from:
        last_epoch, (best_metric, test_metric) = load(
            args.checkpoint_path_from, model, optimizer, scheduler
        )
        logger.info(
            f"Reloaded from {args.checkpoint_path_from}: Last Epoch {last_epoch}, Best Metric {best_metric}, Test Metric {test_metric}"
        )

    metrics = compute_scores

    # Training phase
    if args.phase == "TRAIN_DETECTION" or args.phase == "PRETRAIN_VIT":
        if args.checkpoint_path_from:
            _, _ = load(args.checkpoint_path_from, model, optimizer, scheduler)

        criterion = None
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch: {epoch}")
            train_loss = train(
                args,
                train_loader,
                model,
                optimizer,
                criterion,
                args.epochs,
                epoch,
                scheduler=scheduler,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                scaler=scaler,
                writer=writer,
            )
            if args.phase == "TRAIN_DETECTION":
                test_loss, result = test_detection(
                    args=args,
                    model=model.object_detector,
                    data_loader=test_loader,
                    logger=logger,
                    epoch=epoch,
                    writer=writer,
                )
                save_path = os.path.join(
                    args.checkpoint_path_to,
                    f'epoch_{epoch}_{result["overall_metrics"]["mAP"]}.pth',
                )

            elif args.phase == "PRETRAIN_VIT":
                test_loss, result = test_vit(
                    args=args,
                    model=model,
                    data_loader=test_loader,
                    logger=logger,
                    mode="test",
                    epoch=epoch,
                    writer=writer,
                )
                save_path = os.path.join(
                    args.checkpoint_path_to,
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
    elif args.phase == "INFER_BERT":
        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch: {epoch}")
            train_loss = infer_bert(
                args,
                train_loader,
                model,
                num_epochs=args.epochs,
                current_epoch=epoch,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
            )

    elif args.mode == "TEST":
        # 确保提供了checkpoint路径
        if not args.checkpoint_path_from:
            raise ValueError("必须提供checkpoint路径用于测试!")

        # 加载模型权重
        _, _ = load(args.checkpoint_path_from, model, optimizer, scheduler)
        logger.info(f"从 {args.checkpoint_path_from} 加载模型权重")

        # 保存生成结果
        save_generations(
            args,
            test_loader,
            model,
            logger,
            save_dir=os.path.join(args.checkpoint_path_to, "generations"),
            mode="test",
            device="cuda",
            kw_src=args.kw_src,
            kw_tgt=args.kw_tgt,
        )

    else:
        raise ValueError("Invalid phase")
