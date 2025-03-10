import torch
import torch.nn as nn

class MOE(nn.Module):
    def __init__(
        self,
        args,
        object_detector=None,
        image_encoder=None,
        history_encoder=None,
        modality_fusion=None,
        findings_decoder=None,
        cxr_bert_feature_extractor=None
    ):
        super(MOE, self).__init__()
        
        # 初始化各个组件
        self.object_detector = object_detector
        self.image_encoder = image_encoder
        self.history_encoder = history_encoder
        self.modality_fusion = modality_fusion
        self.findings_decoder = findings_decoder
        self.cxr_bert_feature_extractor = cxr_bert_feature_extractor
        
        # 保存参数配置
        self.args = args

    def forward(
        self,
        image,
        bbox_targets=None,
        findings=None,
        history=None,
        targets=None,
        train_stage=1,
        mode="train",
    ):
        # 在这里实现前向传播逻辑
        if train_stage == 1:
            return self.object_detector(image, bbox_targets)


