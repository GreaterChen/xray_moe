"""
默认配置文件，包含所有可配置项的默认值
不要直接修改此文件，而是创建一个local_config.py来覆盖需要修改的配置项
"""


class DefaultConfig:
    # Debug模式
    DEBUG = True

    # 数据目录设置
    ROOT_DIR = "/path/to/xray_moe/"  # 需要在local_config中覆盖
    DATA_DIR = "/path/to/MIMIC/"  # 需要在local_config中覆盖
    ANN_DIR = "/path/to/mimic_annotation_moe_bbox_filtered_split_numeric.json"  # 需要在local_config中覆盖
    NEGATIVE_POOL_DIR = "/path/to/pool.npy"  # 需要在local_config中覆盖

    # 模型设置
    MODEL_NAME = "MOE"
    IMAGE_SIZE = 224
    DATASET_NAME = "MIMIC"
    MAX_LEN_FINDINGS = 100
    MAX_LEN_HISTORY = 50
    TOKENIZER_MAX_LEN = 30523
    NUM_DISEASES = 14  # 疾病类别数量
    TEMPERATURE = 0.07  # 对比学习温度参数

    # 输入输出关键字
    KW_SRC = ["image", "findings", "history", "bbox_targets"]
    KW_TGT = ["findings", "label"]

    # 训练设置
    PHASE = "FINETUNE_LLAMA"
    MODE = "TRAIN"
    USE_MIXED_PRECISION = True
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 32
    NUM_WORKERS = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WARMUP_LR = 5e-6
    WARMUP_STEPS = 2000
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1

    # 设备设置
    CUDA_VISIBLE_DEVICES = "0"
    SEED = 123

    # 检查点路径
    DETECTION_CHECKPOINT_PATH_FROM = (
        "/path/to/detection_checkpoint.pth"  # 需要在local_config中覆盖
    )
    CHECKPOINT_PATH_FROM = None
    CHECKPOINT_PATH_TO = "/path/to/save/checkpoint/"  # 需要在local_config中覆盖

    # TensorBoard设置
    TENSORBOARD_DIR = "runs"
