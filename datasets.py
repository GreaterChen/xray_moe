# --- Base packages ---
import os
import json
import pickle
import re
import numpy as np
import pandas as pd

# --- PyTorch packages ---
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# --- Helper packages ---
from random import shuffle
import sentencepiece as spm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict
from utils import *


# --- Datasets ---
class MIMIC(data.Dataset):  # MIMIC-CXR Dataset
    # 类变量用于存储共享数据
    _shared_data = {
        "loaded": False,
        "annotation": None,
        "anatomical_embeddings": None,  # 新增：存储解剖区域嵌入数据
    }

    # 29个解剖区域的标准名称映射（按照检测器输出的顺序，从1开始）
    ANATOMICAL_REGIONS = [
        'left hemidiaphragm',      # 1
        'right atrium',            # 2  
        'right hilar structures',  # 3
        'cardiac silhouette',      # 4
        'abdomen',                 # 5
        'trachea',                 # 6
        'right apical zone',       # 7
        'right lung',              # 8
        'right upper lung zone',   # 9
        'right costophrenic angle', # 10
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
        'upper mediastinum'        # 29
    ]

    @classmethod
    def load_anatomical_embeddings(cls, anatomical_db_path):
        """
        加载解剖区域文本嵌入数据
        
        参数:
            anatomical_db_path: 解剖区域数据库文件路径
        """
        if cls._shared_data["anatomical_embeddings"] is not None:
            return  # 已经加载过了
            
        if anatomical_db_path is None or not os.path.exists(anatomical_db_path):
            print(f"⚠️  解剖区域数据库文件不存在或未配置: {anatomical_db_path}")
            cls._shared_data["anatomical_embeddings"] = {}
            return
            
        try:
            print(f"📚 正在加载解剖区域数据库: {anatomical_db_path}")
            with open(anatomical_db_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'image_region_embeddings' in data:
                raw_embeddings = data['image_region_embeddings']
                metadata = data.get('metadata', {})
                print(f"✅ 成功加载解剖区域数据库:")
                print(f"   - 总条目数: {metadata.get('total_keys', len(raw_embeddings))}")
                print(f"   - 向量维度: {metadata.get('embedding_dim', 'Unknown')}")
                print(f"   - 模型名称: {metadata.get('model_name', 'Unknown')}")
                
                # 重新组织数据结构: image_id -> {region_index: tensor}
                organized_embeddings = defaultdict(dict)
                
                for key, embedding in tqdm(raw_embeddings.items(), desc="组织解剖区域数据"):
                    try:
                        # 解析键格式: image_id_region_index
                        parts = key.split('_')
                        if len(parts) >= 2:
                            region_index = int(parts[-1])  # 最后一部分是区域索引
                            image_id = parts[0]  # 前面部分是image_id
                            
                            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                            organized_embeddings[image_id][region_index] = embedding_tensor  # 直接赋值，不append
                    except (ValueError, IndexError):
                        continue  # 忽略格式不正确的键
                
                cls._shared_data["anatomical_embeddings"] = dict(organized_embeddings)
                print(f"📊 组织数据完成，覆盖 {len(organized_embeddings)} 个图像")
                
            else:
                print(f"❌ 数据库格式不正确，缺少 'image_region_embeddings' 字段")
                cls._shared_data["anatomical_embeddings"] = {}
                
        except Exception as e:
            print(f"❌ 加载解剖区域数据库失败: {e}")
            import traceback
            traceback.print_exc()
            cls._shared_data["anatomical_embeddings"] = {}

    @classmethod
    def load_shared_data(cls, directory, ann_dir, mode, extra_ann_dir=None, binary_mode=True):
        """预处理优化版本，加载MIMIC数据集注释"""
        if cls._shared_data["loaded"]:
            return

        # 使用内存映射读取大型JSON文件
        import mmap

        with open(ann_dir, "r") as f:
            # 对大文件使用内存映射
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            annotation_data = json.loads(mm.read().decode("utf-8"))
            mm.close()

        # 并行处理数据拆分
        import concurrent.futures

        new_annotation = {}

        def process_split(mode_split):
            mode, data_split = mode_split
            result = []
            for key, value in data_split.items():
                if value["findings"].strip() != "":
                    value["image_id"] = key
                    # 预处理文本，减少__getitem__中的处理时间
                    value["findings"] = cls._clean_report(value["findings"])
                    value["history"] = cls._clean_report(value["history"])
                    result.append(value)
            return mode, result

        # 并行处理各个拆分
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for mode, result in executor.map(process_split, annotation_data.items()):
                new_annotation[mode] = result

        if extra_ann_dir:
            with open(extra_ann_dir, 'r') as f:
                extra_ann = json.load(f)
            
            for id, item in extra_ann.items():
                item['image_id'] = id
                item['findings'] = cls._clean_report(item['findings'])
                item['history'] = cls._clean_report(item['history'])
                new_annotation['train'].append(item)
                
        cls._shared_data["annotation"] = new_annotation
        cls._shared_data["loaded"] = True



    def __init__(
        self,
        directory,
        ann_dir,
        extra_ann_dir=None,
        input_size=(224, 224),
        random_transform=True,
        tokenizer=None,
        mode="train",
        subset_size=None,
    ):

        self.load_shared_data(directory, ann_dir, mode, extra_ann_dir)

        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.sep_token_id  # BERT使用[SEP]作为EOS
        self.pad_token_id = self.tokenizer.pad_token_id

        self.sources = ["image", "findings", "history", "bbox_targets"]
        self.targets = ["findings", "label"]

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.mode = mode
        self.subset_size = subset_size
        # 使用共享数据
        self.data = self._shared_data["annotation"][self.mode]

        if random_transform:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize(224),
                    # transforms.RandomCrop(input_size),
                    # transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize(224),
                    # transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self):
        if self.subset_size is not None:
            return self.subset_size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        info = self.data[idx]

        findings = info["findings"]
        history = info["history"]
        disease_label = np.array(info["labels"], dtype=np.float16)
        image_id = info['image_id']
        
        # 获取图像路径
        image_base_path = "/".join(info["image_path"][0].split("/")[:-1])
        img_path = os.path.join(
            self.dir, "images_224", image_base_path, info["image_id"] + ".jpg"
        )

        # 处理图像
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 处理bbox
        if "bbox_targets" in info:
            bbox_data = info["bbox_targets"]
            boxes = torch.tensor(bbox_data["boxes"], dtype=torch.float32)
            labels = torch.tensor(bbox_data["labels"], dtype=torch.int64)  # 标签已经是从1开始的

            # 验证并过滤边界框
            valid_boxes = []
            valid_labels = []
            for box, label in zip(boxes, labels):
                # 检查边界框的宽度和高度是否大于0
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width > 0 and height > 0:
                    valid_boxes.append(box)
                    valid_labels.append(label)

            # 如果没有有效的边界框，返回一个空的目标
            if not valid_boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.stack(valid_boxes)
                labels = torch.stack(valid_labels)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,  # [N, 4] tensor，格式为 (x1, y1, x2, y2)
            "labels": labels,  # [N] tensor，已经是从1开始的类别标签
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # 获取该图像的解剖区域文本嵌入（如果有的话）
        anatomical_embeddings = {}
        if self._shared_data["anatomical_embeddings"] and image_id in self._shared_data["anatomical_embeddings"]:
            anatomical_embeddings = self._shared_data["anatomical_embeddings"][image_id]

        output = {
            "image": img,
            "image_id": image_id,
            "bbox_targets": target,
            "findings": findings,
            "history": history,
            "label": disease_label,
            "image_path": img_path,
            "anatomical_embeddings": anatomical_embeddings,  # 新增：该图像的解剖区域嵌入
        }

        return output

    @staticmethod
    def _clean_report(report):
        report_cleaner = (
            lambda t: t.replace("\n", " ")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("1. ", "")
            .replace(". 2. ", ". ")
            .replace(". 3. ", ". ")
            .replace(". 4. ", ". ")
            .replace(". 5. ", ". ")
            .replace(" 2. ", ". ")
            .replace(" 3. ", ". ")
            .replace(" 4. ", ". ")
            .replace(" 5. ", ". ")
            .strip()
            .lower()
            .split(". ")
        )
        sent_cleaner = lambda t: re.sub(
            "[.,?;*!%^&_+():-\[\]{}]",
            "",
            t.replace('"', "")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .strip()
            .lower(),
        )
        tokens = [
            sent_cleaner(sent)
            for sent in report_cleaner(report)
            if sent_cleaner(sent) != []
        ]
        report = " . ".join(tokens) + " ."
        return report


# 添加collate_fn函数处理变长数据
def mimic_collate_fn(batch):
    """优化的collate_fn"""
    # 使用预分配内存而非动态增长的列表
    batch_size = len(batch)

    # 预分配NumPy数组
    images = torch.empty((batch_size, 3, 224, 224), dtype=torch.float32)
    bbox_targets = [None] * batch_size
    findings = [None] * batch_size
    histories = [None] * batch_size
    labels = np.empty((batch_size, 14), dtype=np.float16)  # 假设有14个标签
    image_paths = [None] * batch_size
    image_ids = [None] * batch_size
    anatomical_embeddings = [None] * batch_size  # 新增：解剖区域嵌入

    # 填充预分配的数组
    for i, item in enumerate(batch):
        images[i] = item["image"]
        bbox_targets[i] = item["bbox_targets"]
        findings[i] = item["findings"]
        histories[i] = item["history"]
        labels[i] = item["label"]
        image_paths[i] = item["image_path"]
        image_ids[i] = item["image_id"]
        anatomical_embeddings[i] = item["anatomical_embeddings"]  # 新增

    # 转换标签
    label_tensor = torch.from_numpy(labels)

    return {
        "image": images,
        "bbox_targets": bbox_targets,
        "findings": findings,
        "history": histories,
        "label": label_tensor,
        "image_path": image_paths,
        "image_id": image_ids,
        "anatomical_embeddings": anatomical_embeddings,  # 新增：解剖区域嵌入
        "gts": (findings, [""]*len(findings)),  # 添加gts字段保持兼容性
        "split": ["train"]*len(findings),  # 添加split字段保持兼容性
    }
