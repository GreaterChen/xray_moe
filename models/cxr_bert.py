import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CXR_BERT_FeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super(CXR_BERT_FeatureExtractor, self).__init__()
        self.device = device
        # 加载预训练的 CXR-BERT 模型和对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)

        self.frozen = False

    def forward(self, texts):
        """
        Args:
            texts: 可以是以下两种格式之一:
                - List[str]: batch_size个生成的文本字符串
                - dict: 已经tokenize过的数据，包含input_ids等键
        Returns:
            cls_embeddings: 文本特征张量，形状为 (B, hidden_size)
            texts: 输入的文本列表或tokenized输入
        """
        if not self.frozen:
            # Freeze all parameters in the BERT model
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()  # 设置模型为评估模式
            self.frozen = True

        # 检查输入类型
        if isinstance(texts, list) and all(isinstance(item, str) for item in texts):
            # 对输入文本进行编码
            inputs = self.tokenizer(
                texts,
                max_length=196,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
        elif hasattr(texts, "input_ids"):
            # 已经tokenize过的数据，直接使用
            inputs = texts
        else:
            raise ValueError("输入必须是字符串列表或包含input_ids的tokenized字典")

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取 [CLS] 标记的嵌入表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings

