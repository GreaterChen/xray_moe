"""
默认配置文件，包含所有可配置项的默认值
不要直接修改此文件，而是创建一个local_config.py来覆盖需要修改的配置项
"""


class DefaultConfig:
    # Debug模式
    DEBUG = False
    
    # 设备设置
    USE_CUDA = False  # 是否使用GPU，设置为False则强制使用CPU
    CUDA_VISIBLE_DEVICES = "0"  # 使用的GPU设备ID，可以是 "0" 或 "0,1,2,3" 等
    USE_DISTRIBUTED = False  # 是否使用分布式训练（多GPU时）

    # 数据目录设置
    ROOT_DIR = "/path/to/xray_moe/"  # 需要在local_config中覆盖
    DATA_DIR = "/path/to/MIMIC/"  # 需要在local_config中覆盖
    ANN_DIR = "/path/to/mimic_annotation_moe_bbox_filtered_split_numeric.json"  # 需要在local_config中覆盖
    IMAGES_DIR = "/path/to/MIMIC/images_224/"  # 需要在local_config中覆盖
    NEGATIVE_POOL_DIR = "/path/to/pool.npy"  # 需要在local_config中覆盖
    SPLIT_CSV_PATH = None  # 数据集划分CSV文件路径（可选，如果不设置则自动划分）
    
    # IU_XRAY数据集设置（用于微调和测试）
    IUXRAY_ANN_PATH = "/mnt/chenlb/IU_XRAY/r2gen_version/annotation_with_history_view_labels_split_multi_view_entries_pa_ap_views.json"
    IUXRAY_IMAGES_DIR = "/mnt/chenlb/IU_XRAY/r2gen_version/images_224"

    # 模型设置
    MODEL_NAME = "MedicalReportGenerator"
    IMAGE_SIZE = 224
    DATASET_NAME = "MIMIC"  # 可选值: "MIMIC", "IUXRAY"
    MAX_LEN_FINDINGS = 150
    MAX_LEN_HISTORY = 50
    TOKENIZER_MAX_LEN = 30523
    NUM_DISEASES = 14  # 疾病类别数量
    TEMPERATURE = 0.07  # 对比学习温度参数
    # 对比损失类型（PRETRAIN_VIT阶段）:
    #   "region": 复杂的区域级别对比学习（考虑NLP状态、同文本区域等）
    #   "clip": 原生CLIP对比学习（image-report层面，配对为正，batch内其他为负）
    #   "simple_region_clip": 简化的区域级CLIP（patch-sentence层面，简单配对定义）
    CONTRASTIVE_LOSS_TYPE = "region"
    
    # 解码器设置
    DECODER_TYPE = "bert"  # 可选值: "bert" 或 "qwen2vl"
    QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # Qwen2.5-VL-3B模型名称或路径
    QWEN_MODEL_PATH = None  # Qwen模型本地路径（可选）
    HF_CACHE_DIR = None  # HuggingFace缓存根目录（可选）
    HF_LOCAL_FILES_ONLY = False  # 是否仅使用本地文件（离线模式）
    USE_HISTORY = False  # 是否在解码器中使用历史文本作为prompt
    
    # LoRA微调设置（仅用于Qwen2.5-VL decoder）
    USE_LORA = True  # 是否使用LoRA微调（推荐开启以节省显存）
    LORA_R = 8  # LoRA rank（越大模型容量越大，但显存占用也越多）
    LORA_ALPHA = 16  # LoRA alpha（通常设为 r 的 2倍）
    LORA_DROPOUT = 0.05  # LoRA dropout
    
    # 生成目标设置
    GENERATION_TARGET = "all"  # 可选值: "findings" 或 "all" (findings + impression)
    
    # 目标检测评估设置
    DETECTION_CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
    DETECTION_IOU_THRESHOLDS = [0.3, 0.5, 0.75, 0.9]  # 用于评估的IoU阈值列表

    # 区域级别对比学习设置
    ANATOMICAL_DATABASE_PATH = None  # 解剖区域知识库路径，需要在local_config中设置
    REGION_ITC_TEMPERATURE = 0.07  # 区域级别ITC的温度参数
    ENABLE_REGION_ITC = True  # 是否启用区域级别的ITC损失
    REGION_ITC_WEIGHT = 1.0  # 区域级别ITC损失的权重

    # 输入输出关键字
    KW_SRC = ["image", "findings", "history", "bbox_targets"]
    KW_TGT = ["findings", "label"]

    # 训练设置
    PHASE = "INFER_BERT"
    MODE = "TRAIN"
    USE_MIXED_PRECISION = True
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 16
    NUM_WORKERS = 8
    EPOCHS = 50
    LEARNING_RATE = 5e-5  # 降低学习率，更适合BERT微调
    MIN_LR = 1e-6
    WARMUP_LR = 5e-6
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.001  # 降低weight decay
    DROPOUT = 0.1
    GRAD_CLIP_NORM = 1.0  # 梯度裁剪阈值

    # 分层学习率设置（用于FINETUNE_BERT阶段的参数分组优化）
    USE_LAYERWISE_LR = True  # 启用分层学习率，BERT微调应该使用不同学习率
    BERT_LR_SCALE = 0.1  # BERT参数学习率更低（相对于LEARNING_RATE）
    VIT_LR_SCALE = 0.5  # ViT参数学习率中等
    OTHER_LR_SCALE = 1.0  # 其他参数（投影层、RGAT等）使用标准学习率

    # 随机种子
    SEED = 123
    
    # 多卡并行设置（当USE_CUDA=True且有多个GPU时使用）
    MULTI_GPU_STRATEGY = "auto"  # 多GPU策略: "auto", "dp"(DataParallel), "ddp"(DistributedDataParallel)
    ADJUST_LR_FOR_MULTI_GPU = True  # 是否根据GPU数量自动调整学习率
    SYNC_BN = False  # 是否使用同步BatchNorm（仅在DDP模式下有效）

    # 检查点路径
    DETECTION_CHECKPOINT_PATH_FROM = "/path/to/detection_checkpoint.pth"  # 需要在local_config中覆盖
    CHECKPOINT_PATH_FROM = None
    CHECKPOINT_PATH_TO = "/path/to/save/checkpoint/"  # 需要在local_config中覆盖
    VIT_MODEL_NAME = "google/vit-base-patch16-224"  # ViT模型名称或路径
    VIT_MODEL_PATH = None  # ViT模型本地路径（可选）
    VIT_CACHE_DIR = None  # ViT专用缓存目录（默认使用HF_CACHE_DIR）
    VIT_LOCAL_FILES_ONLY = False  # 是否仅使用本地ViT模型文件
    VIT_CHECKPOINT_PATH_FROM = None  # ViT预训练权重路径（用于微调阶段）
    DECODER_CHECKPOINT_PATH_FROM = None  # 解码器权重路径（用于继续训练）
    IMAGE_ENCODER_CHECKPOINT_PATH_FROM = None  # 图像编码器权重路径（用于推理）

    # TensorBoard设置
    TENSORBOARD_DIR = "runs"
    
    # 评估设置
    EVAL_FREQ = 1  # 每N个epoch评估一次
    
    # CheXbert路径
    CHEXBERT_CHECKPOINT_PATH = None  # CheXbert模型路径，需要在local_config中设置
    BERT_PRETRAINED_PATH = "bert-base-uncased"  # BERT预训练模型路径（用于CheXbert tokenizer和config），可以是Hugging Face模型名或本地路径
    
    # BERT微调模型选择（医学领域模型效果更好）
    # 推荐医学模型:
    # - "emilyalsentzer/Bio_ClinicalBERT" (最推荐，MIMIC-III临床笔记预训练)
    # - "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12" (PubMed + MIMIC)
    # - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" (PubMed预训练)
    # - "allenai/scibert_scivocab_uncased" (科学文献预训练)
    # 通用模型:
    # - "bert-base-uncased" (原始BERT)
    # - "roberta-base" (RoBERTa，性能更好)
    BERT_PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"  # 可在local_config中改为医学模型
    
    # RGAT (Relational Graph Attention Network) 配置
    ENABLE_RGAT = True  # 是否启用RGAT模块（如果为False，decoder将只使用视觉特征）
    AA_ADJ_PATH = None  # Anatomy-Anatomy邻接矩阵路径 (需要在local_config中设置)
    DD_ADJ_PATH = None  # Disease-Disease邻接矩阵路径 (需要在local_config中设置)
    DA_ADJ_PATH = None  # Disease-Anatomy邻接矩阵路径 (需要在local_config中设置)
    RGAT_DROPOUT = 0.1  # RGAT模块的dropout率
    RGAT_LOSS_WEIGHT = 1.0  # RGAT疾病分类损失的权重
    ENABLE_RGAT_CLASSIFICATION_LOSS = True  # 是否启用RGAT分类损失分支

    # 编码器类型: "detection+vit"（现有方案）或 "vit_only"（新增方案, 不经目标检测，直接ViT编码）
    ENCODER_TYPE = "detection+vit"