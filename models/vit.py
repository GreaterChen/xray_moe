import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig


def _has_required_files(dir_path, candidate_files):
    for filename in candidate_files:
        if os.path.exists(os.path.join(dir_path, filename)):
            return True
    return False


def resolve_local_hf_path(path, candidate_files=None, logger=None):
    """
    解析本地HuggingFace缓存路径，必要时自动定位snapshots子目录
    """
    if not path:
        return path
    
    if isinstance(candidate_files, str):
        candidate_files = (candidate_files,)
    candidate_files = candidate_files or ("config.json",)
    
    if os.path.isfile(path):
        return path
    
    if os.path.isdir(path):
        if _has_required_files(path, candidate_files):
            return path
        
        snapshots_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshot_dirs = []
            for name in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, name)
                if os.path.isdir(candidate):
                    try:
                        mtime = os.path.getmtime(candidate)
                    except OSError:
                        mtime = 0
                    snapshot_dirs.append((mtime, candidate))
            for _, candidate in sorted(snapshot_dirs, key=lambda x: x[0], reverse=True):
                if _has_required_files(candidate, candidate_files):
                    if logger:
                        logger.info(f"检测到本地ViT snapshot路径: {candidate}")
                    return candidate
        if logger:
            logger.warning(f"未在路径 {path} 找到 {candidate_files}，将按原路径尝试加载。")
    
    return path


class MedicalVisionTransformer(nn.Module):
    """
    标准医学视觉Transformer模型
    用于提取解剖区域的视觉特征,只负责特征编码
    """

    def __init__(
        self,
        config=None,
        pretrained_vit_name="google/vit-base-patch16-224",
        num_regions=29,
    ):
        super(MedicalVisionTransformer, self).__init__()
        
        if config is None:
            try:
                from configs import config as global_config
            except ImportError:
                global_config = None
            config = global_config
        self.config_obj = config
        
        default_model_name = pretrained_vit_name
        vit_model_name = default_model_name
        vit_model_path = None
        vit_cache_dir = None
        vit_local_files_only = False
        
        if self.config_obj is not None:
            vit_model_name = getattr(self.config_obj, 'VIT_MODEL_NAME', default_model_name)
            vit_model_path = getattr(self.config_obj, 'VIT_MODEL_PATH', None)
            vit_cache_dir = getattr(self.config_obj, 'VIT_CACHE_DIR', None)
            vit_local_files_only = getattr(
                self.config_obj,
                'VIT_LOCAL_FILES_ONLY',
                getattr(self.config_obj, 'HF_LOCAL_FILES_ONLY', False)
            )
            if vit_cache_dir is None:
                vit_cache_dir = getattr(self.config_obj, 'HF_CACHE_DIR', None)
        else:
            vit_model_path = None
            vit_cache_dir = None
        
        model_source = vit_model_path or vit_model_name
        resolved_model_source = resolve_local_hf_path(
            model_source,
            candidate_files=("config.json",),
        )
        
        load_kwargs = {}
        if vit_cache_dir is not None:
            load_kwargs["cache_dir"] = vit_cache_dir
        
        if vit_local_files_only or os.path.exists(resolved_model_source):
            load_kwargs["local_files_only"] = True
        
        self.model_source = resolved_model_source

        # 加载预训练ViT配置
        self.config = ViTConfig.from_pretrained(self.model_source, **load_kwargs)
        self.hidden_size = self.config.hidden_size

        # 创建可学习的 [CLS] token（用于对比学习）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # 使用预训练ViT中的encoder部分
        pretrained_vit = ViTModel.from_pretrained(self.model_source, **load_kwargs)
        self.encoder = pretrained_vit.encoder

        # 初始化最终输出的归一化层 - 与原始ViT保持一致
        self.layernorm = nn.LayerNorm(self.hidden_size)

        # 初始化cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # 保存区域数量
        self.num_regions = num_regions

    def forward(self, region_features):
        """
        标准ViT前向传播
        
        Args:
            region_features: [batch_size, num_regions, hidden_size] - 区域特征
            
        Returns:
            visual_features: [batch_size, 1+num_regions, hidden_size] - 包含CLS token的完整视觉特征
        """
        batch_size = region_features.shape[0]
        
        # 扩展并添加CLS token到区域特征前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 1+num_regions, 768]
        
        # 一次性通过ViT Encoder（不再手动逐层循环）
        # 兼容不同Transformers版本的返回签名
        encoder_outputs = self.encoder(x)
        hidden_states = encoder_outputs.last_hidden_state
        
        # 使用全局LayerNorm处理最终输出
        final_hidden_states = self.layernorm(hidden_states)
        
        return final_hidden_states  # [B, 1+num_regions, hidden_size]