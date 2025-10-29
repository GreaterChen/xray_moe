"""
统一的常量定义
包括解剖区域和疾病的标准顺序

注意：
- 解剖区域在代码中索引从0-28（列表索引）
- 但在bbox_targets的label中是1-29（1是left hemidiaphragm，29是upper mediastinum，0是背景）
- 图矩阵CSV文件已按此顺序排列，直接加载即可
"""

# 解剖区域标准顺序（共29个，列表索引0-28，对应bbox label 1-29）
ANATOMY_ORDER = [
    'left hemidiaphragm',      # 1
    'right atrium',            # 2
    'right hilar structures',  # 3
    'cardiac silhouette',      # 4
    'abdomen',                 # 5
    'trachea',                 # 6
    'right apical zone',       # 7
    'right lung',              # 8
    'right upper lung zone',   # 9
    'right costophrenic angle',# 10
    'svc',                     # 11
    'left lung',               # 12
    'right mid lung zone',     # 13
    'cavoatrial junction',     # 14
    'left costophrenic angle', # 15
    'left hilar structures',   # 16
    'mediastinum',             # 17
    'right lower lung zone',   # 18
    'left mid lung zone',      # 19
    'spine',                   # 20
    'left upper lung zone',    # 21
    'right hemidiaphragm',     # 22
    'left clavicle',           # 23
    'aortic arch',             # 24
    'right clavicle',          # 25
    'left apical zone',        # 26
    'left lower lung zone',    # 27
    'carina',                  # 28
    'upper mediastinum',       # 29
]

# 疾病标准顺序（共14个，列表索引0-13）
DISEASE_ORDER = [
    'Atelectasis',                  # 0
    'Cardiomegaly',                 # 1
    'Consolidation',                # 2
    'Edema',                        # 3
    'Enlarged Cardiomediastinum',   # 4
    'Fracture',                     # 5
    'Lung Lesion',                  # 6
    'Lung Opacity',                 # 7
    'No Finding',                   # 8
    'Pleural Effusion',             # 9
    'Pleural Other',                # 10
    'Pneumonia',                    # 11
    'Pneumothorax',                 # 12
    'Support Devices'               # 13
]

# 数量常量
NUM_ANATOMY_REGIONS = len(ANATOMY_ORDER)  # 29
NUM_DISEASES = len(DISEASE_ORDER)  # 14

# 创建名称到索引的映射（便于快速查找）
ANATOMY_TO_IDX = {name: idx for idx, name in enumerate(ANATOMY_ORDER)}  # 0-28
DISEASE_TO_IDX = {name: idx for idx, name in enumerate(DISEASE_ORDER)}  # 0-13

# bbox label到列表索引的映射（bbox label: 1-29，列表索引: 0-28）
BBOX_LABEL_TO_IDX = {i+1: i for i in range(NUM_ANATOMY_REGIONS)}  # {1:0, 2:1, ..., 29:28}

# 列表索引到bbox label的映射
IDX_TO_BBOX_LABEL = {i: i+1 for i in range(NUM_ANATOMY_REGIONS)}  # {0:1, 1:2, ..., 28:29}

