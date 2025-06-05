import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import MIMIC


class AnatomicalTextEnhancer(nn.Module):
    """
    解剖区域文本增强模块
    为每个视觉token找到相似度最高的解剖区域文本token
    """
    
    def __init__(self, tokenizer, visual_projection=None, text_projection=None, device="cuda"):
        super(AnatomicalTextEnhancer, self).__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        self.visual_projection = visual_projection
        self.text_projection = text_projection
        
        # 获取解剖区域数据
        self.anatomical_data = MIMIC.get_anatomical_embeddings()
        self.region_mapping = MIMIC.get_region_mapping()
        
        if self.anatomical_data is None:
            print("警告: 未加载解剖区域embeddings，文本增强功能将被禁用")
            self.enabled = False
            return
        
        self.enabled = True
        
        # 预处理解剖区域embeddings，转换为tensor并移到设备上
        self.region_embeddings = {}
        self.region_phrases = {}
        
        for region_idx, region_name in self.region_mapping.items():
            if region_name is not None and region_name in self.anatomical_data['embeddings']:
                # 转换embeddings为tensor
                embeddings = torch.tensor(
                    self.anatomical_data['embeddings'][region_name], 
                    dtype=torch.float32,
                    device=device
                )
                self.region_embeddings[region_idx] = embeddings  # [n_phrases, 768]
                
                # 存储对应的句子
                self.region_phrases[region_idx] = self.anatomical_data['phrases'][region_name]
        
        print(f"文本增强模块初始化完成，支持 {len(self.region_embeddings)} 个解剖区域")
        if self.visual_projection is not None and self.text_projection is not None:
            print("启用共同空间映射进行相似度计算")
    
    def find_similar_texts(self, visual_features):
        """
        为每个视觉token找到最相似的文本
        
        Args:
            visual_features: [batch_size, num_regions+1, hidden_size] ViT输出特征
                           第0个token是CLS，第1-29个token是解剖区域
        
        Returns:
            enhanced_texts: list of lists，每个样本对应一个文本列表
            similarity_scores: [batch_size, num_regions] 相似度分数
        """
        if not self.enabled:
            return None, None
        
        batch_size = visual_features.size(0)
        num_regions = 29
        
        # 提取区域特征（跳过CLS token）
        region_features = visual_features[:, 1:1+num_regions, :]  # [batch_size, 29, 768]
        
        enhanced_texts = []
        all_similarity_scores = []
        
        for batch_idx in range(batch_size):
            batch_texts = []
            batch_similarities = []
            
            for region_idx in range(num_regions):
                if region_idx in self.region_embeddings:
                    # 获取当前区域的视觉特征
                    visual_feat = region_features[batch_idx, region_idx]  # [768]
                    
                    # 获取当前区域的所有文本embeddings
                    text_embeddings = self.region_embeddings[region_idx]  # [n_phrases, 768]
                    
                    # 如果有projection层，先映射到共同空间
                    if self.visual_projection is not None and self.text_projection is not None:
                        # 映射视觉特征到共同空间
                        mapped_visual_feat = self.visual_projection(visual_feat)  # [768]
                        
                        # 映射文本embeddings到共同空间
                        mapped_text_embeddings = self.text_projection(text_embeddings)  # [n_phrases, 768]
                        
                        # 在映射后的空间中计算余弦相似度
                        visual_feat_norm = F.normalize(mapped_visual_feat.unsqueeze(0), p=2, dim=1)  # [1, 768]
                        text_embeddings_norm = F.normalize(mapped_text_embeddings, p=2, dim=1)  # [n_phrases, 768]
                    else:
                        # 如果没有projection层，直接在原始空间计算相似度
                        visual_feat_norm = F.normalize(visual_feat.unsqueeze(0), p=2, dim=1)  # [1, 768]
                        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=1)  # [n_phrases, 768]
                    
                    similarities = torch.mm(visual_feat_norm, text_embeddings_norm.t())  # [1, n_phrases]
                    similarities = similarities.squeeze(0)  # [n_phrases]
                    
                    # 找到最相似的文本
                    best_idx = torch.argmax(similarities).item()
                    best_similarity = similarities[best_idx].item()
                    best_text = self.region_phrases[region_idx][best_idx]
                    
                    batch_texts.append(best_text)
                    batch_similarities.append(best_similarity)
                else:
                    # 如果该区域没有文本数据，使用默认文本
                    batch_texts.append("")
                    batch_similarities.append(0.0)
            
            enhanced_texts.append(batch_texts)
            all_similarity_scores.append(batch_similarities)
        
        # 转换相似度分数为tensor
        similarity_scores = torch.tensor(all_similarity_scores, device=self.device)  # [batch_size, 29]
        
        return enhanced_texts, similarity_scores
    
    def create_enhanced_prompt(self, enhanced_texts, history_text=None, similarity_threshold=0.3):
        """
        创建增强的提示文本
        
        Args:
            enhanced_texts: list of lists，每个样本的增强文本列表
            history_text: 原始历史文本（可选）
            similarity_threshold: 相似度阈值，低于此值的文本将被过滤
        
        Returns:
            enhanced_prompts: list of str，每个样本的增强提示文本
        """
        if not self.enabled or enhanced_texts is None:
            return [history_text] * len(enhanced_texts) if history_text else [""] * len(enhanced_texts)
        
        enhanced_prompts = []
        
        for batch_idx, (texts, similarities) in enumerate(zip(enhanced_texts, self.all_similarity_scores)):
            # 过滤低相似度的文本
            valid_texts = []
            for text, sim_score in zip(texts, similarities):
                if sim_score >= similarity_threshold and text.strip():
                    valid_texts.append(text.strip())
            
            # 构建增强提示
            if valid_texts:
                # 去重并限制长度
                unique_texts = list(dict.fromkeys(valid_texts))  # 保持顺序的去重
                enhanced_text = " . ".join(unique_texts[:10])  # 最多使用10个最相似的文本
                
                # 与历史文本结合
                if history_text and isinstance(history_text, list):
                    hist_text = history_text[batch_idx] if batch_idx < len(history_text) else ""
                elif history_text and isinstance(history_text, str):
                    hist_text = history_text
                else:
                    hist_text = ""
                
                if hist_text.strip():
                    prompt = f"{hist_text.strip()} [SEP] Anatomical findings: {enhanced_text}"
                else:
                    prompt = f"Anatomical findings: {enhanced_text}"
            else:
                # 如果没有有效的增强文本，使用原始历史文本
                if history_text and isinstance(history_text, list):
                    prompt = history_text[batch_idx] if batch_idx < len(history_text) else ""
                elif history_text and isinstance(history_text, str):
                    prompt = history_text
                else:
                    prompt = ""
            
            enhanced_prompts.append(prompt)
        
        return enhanced_prompts
    
    def forward(self, visual_features, history_text=None, similarity_threshold=0.3):
        """
        完整的文本增强流程
        
        Args:
            visual_features: [batch_size, num_regions+1, hidden_size] ViT输出特征
            history_text: 原始历史文本
            similarity_threshold: 相似度阈值
        
        Returns:
            enhanced_prompts: list of str，增强后的提示文本
            similarity_scores: [batch_size, num_regions] 相似度分数
        """
        if not self.enabled:
            batch_size = visual_features.size(0)
            if history_text and isinstance(history_text, list):
                return history_text, torch.zeros((batch_size, 29), device=self.device)
            else:
                return [history_text or ""] * batch_size, torch.zeros((batch_size, 29), device=self.device)
        
        # 找到相似文本
        enhanced_texts, similarity_scores = self.find_similar_texts(visual_features)
        
        # 保存相似度分数以供create_enhanced_prompt使用
        self.all_similarity_scores = similarity_scores.cpu().numpy().tolist()
        
        # 创建增强提示
        enhanced_prompts = self.create_enhanced_prompt(
            enhanced_texts, 
            history_text, 
            similarity_threshold
        )
        
        return enhanced_prompts, similarity_scores 