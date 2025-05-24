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
    }

    @classmethod
    def load_shared_data(cls, directory, ann_dir, mode, binary_mode=True):
        """预处理优化版本"""
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

        cls._shared_data["annotation"] = new_annotation
        cls._shared_data["loaded"] = True

    def __init__(
        self,
        directory,
        ann_dir,
        input_size=(224, 224),
        random_transform=True,
        tokenizer=None,
        mode="train",
        subset_size=None,
    ):

        self.load_shared_data(directory, ann_dir, mode)

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

        output = {
            "image": img,
            "bbox_targets": target,
            "findings": findings,
            "history": history,
            "label": disease_label,
            "image_path": img_path,
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

    # 填充预分配的数组
    for i, item in enumerate(batch):
        images[i] = item["image"]
        bbox_targets[i] = item["bbox_targets"]
        findings[i] = item["findings"]
        histories[i] = item["history"]
        labels[i] = item["label"]
        image_paths[i] = item["image_path"]

    # 转换标签
    label_tensor = torch.from_numpy(labels)

    return {
        "image": images,
        "bbox_targets": bbox_targets,
        "findings": findings,
        "history": histories,
        "label": label_tensor,
        "image_path": image_paths,
    }
