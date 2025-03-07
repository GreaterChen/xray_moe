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

        filtered_annotation = {}
        for key, data_split in annotation_data.items():
            filtered_annotation[key] = [
                item for item in data_split if item["findings"].strip() != ""
            ]
        cls._shared_data["annotation"] = filtered_annotation
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

        self.sources = ["image", "findings", "impression", "history"]
        self.targets = ["findings", "impression", "label"]

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
                    transforms.Resize(256),
                    transforms.RandomCrop(input_size),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(input_size),
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

        impression = self.my_pre_caption(info["impression"])
        findings = self.my_pre_caption(info["findings"])
        history = self.my_pre_caption(info["history"])
        findings = f"findings {findings} impression {impression}"
        label = np.array(info["labels"], dtype=np.float32)

        # 获取图像路径
        img_path = os.path.join(self.dir, "images", info["image_path"][0])

        output = {
            "findings": findings,
            "impression": impression,
            "history": history,
            "label": label,
            "gts": [findings, impression],
            "image_path": img_path,  # 添加完整图像路径
            "split": self.mode,  # 添加数据集划分信息
        }

        # 处理图像
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        output["image"] = img

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
