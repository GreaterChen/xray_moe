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

    # 模型设置
    MODEL_NAME = "MOE"
    IMAGE_SIZE = 224
    DATASET_NAME = "MIMIC"
    MAX_LEN_FINDINGS = 100
    MAX_LEN_HISTORY = 50
    TOKENIZER_MAX_LEN = 30523
    NUM_DISEASES = 14  # 疾病类别数量
    TEMPERATURE = 0.07  # 对比学习温度参数
    
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
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WARMUP_LR = 5e-6
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1

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
    VIT_CHECKPOINT_PATH_FROM = None  # ViT预训练权重路径（用于微调阶段）
    DECODER_CHECKPOINT_PATH_FROM = None  # 解码器权重路径（用于继续训练）
    IMAGE_ENCODER_CHECKPOINT_PATH_FROM = None  # 图像编码器权重路径（用于推理）

    # TensorBoard设置
    TENSORBOARD_DIR = "runs"
    
    # 评估设置
    EVAL_FREQ = 5  # 每N个epoch评估一次
    
    # CheXbert路径
    CHEXBERT_CHECKPOINT_PATH = None  # CheXbert模型路径，需要在local_config中设置

    # 文本增强模块配置
    ENABLE_TEXT_ENHANCEMENT = False  # 是否启用文本增强功能
    TEXT_ENHANCEMENT_PHASES = ["FINETUNE_BERT"]  # 支持文本增强的训练阶段
    TEXT_ENHANCEMENT_DB_PATH = None  # 文本增强数据库路径，需要在local_config中设置
    TEXT_ENHANCEMENT_SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
    TEXT_ENHANCEMENT_TOP_K = 1  # 每个区域返回的top相似样本数
    TEXT_ENHANCEMENT_TOP_SENTENCES = 5  # 全局选择的top句子数（仅用于传统文本拼接方式）
    
    # Cross-Attention文本增强配置
    TEXT_ENHANCEMENT_USE_CROSS_ATTENTION = False  # 是否使用Cross-Attention方式（新方法）
    TEXT_ENHANCEMENT_CROSS_ATTN_HEADS = 12  # Cross-Attention头数
    TEXT_ENHANCEMENT_CROSS_ATTN_DROPOUT = 0.1  # Cross-Attention dropout率
    TEXT_ENHANCEMENT_FUSION_WEIGHT = 0.3  # 文本增强特征的固定融合权重
