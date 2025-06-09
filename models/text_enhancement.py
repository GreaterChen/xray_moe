import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from datasets import MIMIC
from collections import defaultdict


class AnatomicalTextEnhancer(nn.Module):
    """
    解剖区域文本增强模块 - 基于视觉特征检索
    通过视觉特征检索相似的视觉特征，然后获取对应的文本描述
    支持按解剖区域分组检索和排除自身样本
    """
    
    def __init__(self, tokenizer, visual_projection=None, text_projection=None, device="cuda"):
        super(AnatomicalTextEnhancer, self).__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        self.visual_projection = visual_projection
        self.text_projection = text_projection
        
        # 加载新的视觉-文本知识库
        self.knowledge_base_path = "/mnt/chenlb/datasets/MIMIC/anatomical_database/visual_text_knowledge_base.pkl"
        
        if not os.path.exists(self.knowledge_base_path):
            print(f"警告: 知识库文件 {self.knowledge_base_path} 不存在，文本增强功能将被禁用")
            self.enabled = False
            return
        
        try:
            # 加载知识库
            with open(self.knowledge_base_path, 'rb') as f:
                kb_data = pickle.load(f)
            
            self.knowledge_base = kb_data['knowledge_base']
            self.feature_to_info_map = kb_data['feature_to_info_map']
            self.statistics = kb_data['statistics']
            
            # 获取解剖区域列表（与MIMIC.ANATOMICAL_REGIONS对应）
            anatomical_regions = MIMIC.ANATOMICAL_REGIONS
            region_name_to_idx = {name: idx for idx, name in enumerate(anatomical_regions)}
            
            # 直接按解剖区域组织知识库，避免重复存储
            self.region_features_db = {}  # 按区域存储的特征张量
            self.region_indices = {}      # 按区域存储的原始索引
            self.region_image_ids = {}    # 按区域存储的image_id
            self.region_texts = {}        # 按区域存储的文本描述
            
            # 预分配存储结构
            region_data_temp = {idx: {'features': [], 'indices': [], 'image_ids': [], 'texts': []} 
                               for idx in range(len(anatomical_regions))}
            
            # 一次遍历组织所有数据
            for idx, entry in enumerate(self.knowledge_base):
                region_name = entry['region_name']
                if region_name in region_name_to_idx:
                    region_idx = region_name_to_idx[region_name]
                    region_data_temp[region_idx]['features'].append(entry['visual_feature'])
                    region_data_temp[region_idx]['indices'].append(idx)
                    region_data_temp[region_idx]['image_ids'].append(entry['image_id'])
                    region_data_temp[region_idx]['texts'].append(entry['text_string'])
            
            # 转换为tensor并归一化，只处理有数据的区域
            successful_regions = 0
            for region_idx, region_name in enumerate(anatomical_regions):
                region_data = region_data_temp[region_idx]
                
                if region_data['features']:  # 确保不为空
                    # 批量转换和归一化
                    features_array = np.stack(region_data['features'])
                    region_features_tensor = torch.tensor(
                        features_array, 
                        dtype=torch.float32, 
                        device=device
                    )
                    region_features_tensor = F.normalize(region_features_tensor, p=2, dim=1)
                    
                    self.region_features_db[region_idx] = region_features_tensor
                    self.region_indices[region_idx] = region_data['indices']
                    self.region_image_ids[region_idx] = region_data['image_ids']
                    self.region_texts[region_idx] = region_data['texts']
                    successful_regions += 1
                else:
                    print(f"警告: 解剖区域 '{region_name}' (索引{region_idx}) 没有数据")
            
            # 清理临时数据
            del region_data_temp
            del self.knowledge_base  # 释放原始知识库内存
            
            self.enabled = True
            
            print(f"视觉特征知识库加载成功:")
            print(f"  - 总条目数: {self.statistics['total_entries']}")
            print(f"  - 覆盖图像数: {self.statistics['unique_images']}")
            print(f"  - 解剖区域数: {self.statistics['unique_regions']}")
            print(f"  - 成功组织的区域数: {successful_regions}")
            
            # 打印每个区域的数据量
            for region_idx, region_name in enumerate(anatomical_regions):
                if region_idx in self.region_features_db:
                    count = len(self.region_texts[region_idx])
                    print(f"    - {region_name}: {count} 个样本")
            
        except Exception as e:
            print(f"加载知识库时出错: {e}")
            self.enabled = False
            return
    
    def retrieve_similar_visual_features(self, query_visual_features, query_image_ids=None, top_k=1):
        """
        基于视觉特征检索相似的视觉特征及其对应的文本描述
        支持按解剖区域分组检索和排除自身样本
        
        Args:
            query_visual_features: [batch_size, num_regions, hidden_size] 查询的视觉特征
            query_image_ids: list of str，查询样本的image_id列表，用于排除自身
            top_k: 返回最相似的top-k个结果
        
        Returns:
            retrieved_texts: list of lists，每个样本每个区域的检索结果文本
            similarity_scores: [batch_size, num_regions] 最高相似度分数
        """
        if not self.enabled:
            return None, None
        
        batch_size, num_regions, hidden_size = query_visual_features.shape
        
        # 如果没有提供image_ids，创建空列表
        if query_image_ids is None:
            query_image_ids = [None] * batch_size
        
        # 归一化查询特征
        query_features_norm = F.normalize(query_visual_features, p=2, dim=2)  # [B, 29, 768]
        
        # 预处理：为每个batch样本预计算有效索引（排除自身image_id）
        batch_valid_indices = []
        for batch_idx in range(batch_size):
            query_image_id = query_image_ids[batch_idx]
            sample_valid_indices = {}
            
            for region_idx in range(num_regions):
                if region_idx not in self.region_features_db:
                    sample_valid_indices[region_idx] = []
                    continue
                
                region_image_ids = self.region_image_ids[region_idx]
                if query_image_id is not None:
                    valid_indices = [
                        i for i, img_id in enumerate(region_image_ids) 
                        if img_id != query_image_id
                    ]
                else:
                    valid_indices = list(range(len(region_image_ids)))
                
                sample_valid_indices[region_idx] = valid_indices
            
            batch_valid_indices.append(sample_valid_indices)
        
        # 批量计算相似度和检索结果
        retrieved_texts = []
        similarity_scores_list = []
        
        for batch_idx in range(batch_size):
            batch_texts = []
            batch_similarities = []
            valid_indices_map = batch_valid_indices[batch_idx]
            
            # 批量处理所有region
            for region_idx in range(num_regions):
                if region_idx not in self.region_features_db:
                    batch_texts.append("")
                    batch_similarities.append(0.0)
                    continue
                
                valid_indices = valid_indices_map[region_idx]
                if not valid_indices:
                    batch_texts.append("")
                    batch_similarities.append(0.0)
                    continue
                
                # 获取查询特征和区域特征
                query_feat = query_features_norm[batch_idx, region_idx]  # [768]
                region_features = self.region_features_db[region_idx]    # [N_region, 768]
                
                # 只对有效索引计算相似度
                valid_region_features = region_features[valid_indices]  # [N_valid, 768]
                
                # 批量计算相似度
                similarities = torch.mm(
                    query_feat.unsqueeze(0),      # [1, 768]
                    valid_region_features.t()     # [768, N_valid]
                ).squeeze(0)  # [N_valid]
                
                # 获取top-k结果
                top_k_actual = min(top_k, len(similarities))
                if top_k_actual == 0:
                    batch_texts.append("")
                    batch_similarities.append(0.0)
                    continue
                
                top_similarities, top_relative_indices = torch.topk(similarities, top_k_actual)
                
                # 转换为原始索引并获取文本
                region_texts = self.region_texts[region_idx]
                if top_k_actual == 1:
                    best_original_idx = valid_indices[top_relative_indices[0].item()]
                    best_similarity = top_similarities[0].item()
                    best_text = region_texts[best_original_idx]
                    
                    batch_texts.append(best_text)
                    batch_similarities.append(best_similarity)
                else:
                    # 多个结果拼接
                    retrieved_descriptions = []
                    for i in range(top_k_actual):
                        original_idx = valid_indices[top_relative_indices[i].item()]
                        retrieved_descriptions.append(region_texts[original_idx])
                    
                    combined_text = " . ".join(retrieved_descriptions)
                    batch_texts.append(combined_text)
                    batch_similarities.append(top_similarities[0].item())
            
            retrieved_texts.append(batch_texts)
            similarity_scores_list.append(batch_similarities)
        
        # 一次性转换相似度分数为tensor
        similarity_scores = torch.tensor(similarity_scores_list, 
                                       dtype=torch.float32, 
                                       device=self.device)  # [B, 29]
        
        return retrieved_texts, similarity_scores
    
    def create_enhanced_prompt(self, retrieved_texts, similarity_scores, history_text=None, similarity_threshold=0.5):
        """
        创建增强的提示文本
        
        Args:
            retrieved_texts: list of lists，每个样本的检索文本列表
            similarity_scores: [batch_size, num_regions] 相似度分数
            history_text: 原始历史文本（可选）
            similarity_threshold: 相似度阈值，低于此值的文本将被过滤
        
        Returns:
            enhanced_prompts: list of str，每个样本的增强提示文本
        """
        if not self.enabled or retrieved_texts is None:
            batch_size = len(retrieved_texts) if retrieved_texts else 1
            if history_text and isinstance(history_text, list):
                return history_text[:batch_size]
            else:
                return [history_text or ""] * batch_size
        
        # 预处理history_text
        is_history_list = isinstance(history_text, list)
        default_history = history_text or ""
        
        enhanced_prompts = []
        batch_size = len(retrieved_texts)
        
        for batch_idx in range(batch_size):
            texts = retrieved_texts[batch_idx]
            similarities = similarity_scores[batch_idx]
            
            # 批量过滤和处理文本
            valid_texts = []
            for text, sim_score in zip(texts, similarities):
                if sim_score >= similarity_threshold:
                    stripped_text = text.strip()
                    if stripped_text:
                        valid_texts.append(stripped_text)
            
            # 构建增强提示
            if valid_texts:
                # 去重并限制长度（使用dict.fromkeys保持顺序）  
                unique_texts = list(dict.fromkeys(valid_texts))[:5]  # 最多5个
                enhanced_text = " . ".join(unique_texts)
                
                # 获取历史文本
                if is_history_list:
                    hist_text = history_text[batch_idx].strip() if batch_idx < len(history_text) else ""
                else:
                    hist_text = default_history.strip() if default_history else ""
                
                # 构建最终提示
                if hist_text:
                    prompt = f"{hist_text} [SEP] Retrieved anatomical findings: {enhanced_text}"
                else:
                    prompt = f"Retrieved anatomical findings: {enhanced_text}"
            else:
                # 没有有效增强文本时使用原始历史文本
                if is_history_list:
                    prompt = history_text[batch_idx] if batch_idx < len(history_text) else ""
                else:
                    prompt = default_history
            
            enhanced_prompts.append(prompt)
        
        return enhanced_prompts
    
    def forward(self, visual_features, history_text=None, query_image_ids=None, similarity_threshold=0.5, top_k=1):
        """
        完整的文本增强流程 - 基于视觉特征检索
        
        Args:
            visual_features: [batch_size, num_regions+1, hidden_size] ViT输出特征
            history_text: 原始历史文本
            query_image_ids: list of str，查询样本的image_id列表，用于排除自身
            similarity_threshold: 相似度阈值
            top_k: 检索的top-k结果数量
        
        Returns:
            enhanced_prompts: list of str，增强后的提示文本
            similarity_scores: [batch_size, num_regions] 相似度分数
        """
        batch_size = visual_features.size(0)
        
        if not self.enabled:
            # 预处理fallback结果
            fallback_scores = torch.zeros((batch_size, 29), device=self.device)
            if history_text:
                if isinstance(history_text, list):
                    fallback_prompts = history_text[:batch_size]
                else:
                    fallback_prompts = [history_text] * batch_size
            else:
                fallback_prompts = [""] * batch_size
            return fallback_prompts, fallback_scores
        
        # 提取区域特征（跳过CLS token）
        region_features = visual_features[:, 1:30, :]  # [batch_size, 29, 768]
        
        # 基于视觉特征检索相似的文本描述
        retrieved_texts, similarity_scores = self.retrieve_similar_visual_features(
            region_features, 
            query_image_ids=query_image_ids,
            top_k=top_k
        )
        
        # 创建增强提示
        enhanced_prompts = self.create_enhanced_prompt(
            retrieved_texts, 
            similarity_scores,
            history_text, 
            similarity_threshold
        )
        
        return enhanced_prompts, similarity_scores 