from easydict import EasyDict as edict
import os

# 基础配置
CONFIG = edict()
CONFIG.SEED = 42
CONFIG.DEBUG = False

# 环境配置
CONFIG.CUDA_VISIBLE_DEVICES = "0"
CONFIG.NUM_WORKERS = 8

# 数据配置
CONFIG.DATASET_NAME = "MIMIC"
CONFIG.DATA_DIR = "/path/to/data"
CONFIG.ANN_DIR = "/path/to/annotations"
CONFIG.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG.IMAGE_SIZE = 512
CONFIG.MODE = "train"

# 训练配置
CONFIG.TRAIN_BATCH_SIZE = 4
CONFIG.VAL_BATCH_SIZE = 4
CONFIG.LEARNING_RATE = 1e-4
CONFIG.WEIGHT_DECAY = 1e-4
CONFIG.EPOCHS = 10
CONFIG.MIN_LR = 1e-5
CONFIG.WARMUP_LR = 1e-5
CONFIG.WARMUP_STEPS = 100

# 模型配置
CONFIG.MODEL_NAME = "MOE"
CONFIG.PHASE = "FINETUNE_BERT"  # 使用自定义的BERT解码器阶段

# 目录和路径配置
CONFIG.NEGATIVE_POOL_DIR = "results/negative_samples"
CONFIG.CHECKPOINT_PATH_FROM = "results/detection/model.pth"  # 检测模型的路径
CONFIG.DETECTION_CHECKPOINT_PATH_FROM = "results/detection/model.pth"  # 检测模型的路径
CONFIG.VIT_CHECKPOINT_PATH_FROM = "results/vit/model.pth"  # ViT模型的路径
CONFIG.CHECKPOINT_PATH_TO = "results/bert_decoder"  # 结果保存路径
CONFIG.TENSORBOARD_DIR = "logs/tensorboard"

# KV源配置
CONFIG.KW_SRC = "image history"  # 使用图像和历史文本作为输入
CONFIG.KW_TGT = "findings"  # 生成findings
CONFIG.USE_MIXED_PRECISION = True  # 是否使用混合精度训练

# 添加历史文本和findings文本的最大长度配置
CONFIG.MAX_LEN_HISTORY = 50
CONFIG.MAX_LEN_FINDINGS = 100

config = CONFIG 