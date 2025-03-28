"""
本地配置文件模板
复制此文件为local_config.py，并根据本地环境修改相应的配置项
注意：local_config.py 不应该被提交到git仓库中
"""

from .default_config import DefaultConfig


class LocalConfig(DefaultConfig):
    # 在这里覆盖需要修改的配置项
    ROOT_DIR = "/home/username/xray_moe/"
    DATA_DIR = "/home/username/datasets/MIMIC/"
    ANN_DIR = "/home/username/datasets/MIMIC/mimic_annotation_moe_bbox_filtered_split_numeric.json"
    NEGATIVE_POOL_DIR = "/home/username/xray_moe/results/ltc/negative_pool/pool.npy"

    DETECTION_CHECKPOINT_PATH_FROM = "/home/username/xray_moe/results/ltc/detection/epoch_9_BLEU_1_0.8791605068503925.pth"
    CHECKPOINT_PATH_TO = "/home/username/xray_moe/results/ltc/14_classifier/"
