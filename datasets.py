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
        """静态方法，用于加载所有数据集共享的数据"""
        if cls._shared_data["loaded"]:
            return

        with open(ann_dir, "r") as f:
            annotation_data = json.load(f)
        
        new_annotation = {}
        for mode, data_split in annotation_data.items():
            new_annotation[mode] = []
            for key, value in data_split.items():
                if value["findings"].strip() != "":
                    value["image_id"] = key
                    new_annotation[mode].append(value)

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
                    transforms.Resize(224),
                    # transforms.RandomCrop(input_size),
                    # transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
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

        findings = self.my_pre_caption(info["findings"])
        history = self.my_pre_caption(info["history"])
        disease_label = np.array(info["labels"], dtype=np.float16)
        bbox_data = info["bbox_targets"]

        # 获取图像路径
        image_base_path = '/'.join(info["image_path"][0].split('/')[:-1])
        img_path = os.path.join(self.dir, "images", image_base_path, info["image_id"] + ".jpg")

        # 处理图像
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 处理bbox数据
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

        target = {
            'boxes': boxes,          # [N, 4] tensor，格式为 (x1, y1, x2, y2)
            'labels': labels,        # [N] tensor，已经是从1开始的类别标签
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        output = {
            "image": img,
            "bbox_targets": target,
            "findings": findings,
            "history": history,
            "label": disease_label,
            "image_path": img_path
        }

        return output

    def clean_report_mimic_cxr(self, report):
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

    def my_pre_caption(self, caption, max_words=196):
        caption = self.clean_report_mimic_cxr(caption)
        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:])
        return caption

# 添加collate_fn函数处理变长数据
def mimic_collate_fn(batch):
    """
    自定义collate_fn函数，处理变长的边界框数据
    
    Args:
        batch: 一个批次的数据，每个元素是__getitem__返回的字典
        
    Returns:
        处理后的批次数据，保持字典列表格式
    """
    images = []
    bbox_targets = []
    findings = []
    histories = []
    labels = []
    image_paths = []
    
    for item in batch:
        images.append(item["image"])
        bbox_targets.append(item["bbox_targets"])
        findings.append(item["findings"])
        histories.append(item["history"])
        labels.append(item["label"])
        image_paths.append(item["image_path"])
    
    # 将图像堆叠成批次
    images = torch.stack(images, dim=0)
    labels = torch.from_numpy(np.stack(labels, axis=0))
    
    # 返回字典，其中targets保持为列表形式
    return {
        "image": images,
        "bbox_targets": bbox_targets,  # 保持列表形式，每个元素是一个字典
        "findings": findings,
        "history": histories,
        "label": labels,
        "image_path": image_paths
    }
