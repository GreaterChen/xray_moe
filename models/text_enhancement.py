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
    
    def __init__(self, tokenizer, visual_projection=None, text_projection=None, device="cuda", config=None):
        super(AnatomicalTextEnhancer, self).__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        self.visual_projection = visual_projection
        self.text_projection = text_projection
        
        # 从配置中获取参数
        if config is not None:
            # 检查是否启用文本增强
            self.enabled = getattr(config, 'ENABLE_TEXT_ENHANCEMENT', False)
            
            # 如果未启用，直接返回
            if not self.enabled:
                print("文本增强功能已被配置禁用")
                return
            
            # 获取数据库路径
            self.knowledge_base_path = getattr(config, 'TEXT_ENHANCEMENT_DB_PATH', 
                                             "/mnt/chenlb/datasets/MIMIC/visual_text_knowledge_base.pkl")
        else:
            # 默认配置（向后兼容）
            print("警告: 未提供配置，使用默认设置启用文本增强")
            self.enabled = True
            self.knowledge_base_path = "/mnt/chenlb/datasets/MIMIC/visual_text_knowledge_base.pkl"
        
        # 检查数据库文件是否存在
        if not os.path.exists(self.knowledge_base_path):
            print(f"警告: 知识库文件 {self.knowledge_base_path} 不存在，文本增强功能将被禁用")
            self.enabled = False
            return
        
        try:
            # 加载知识库
            print(f"正在加载文本增强数据库: {self.knowledge_base_path}")
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
            
            print(f"✅ 文本增强数据库加载成功:")
            print(f"  - 数据库路径: {self.knowledge_base_path}")
            print(f"  - 总条目数: {self.statistics['total_entries']}")
            print(f"  - 覆盖图像数: {self.statistics['unique_images']}")
            print(f"  - 解剖区域数: {self.statistics['unique_regions']}")
            print(f"  - 成功组织的区域数: {successful_regions}")
            
            # 打印每个区域的数据量（可选，通过配置控制）
            if hasattr(config, 'SHOW_REGION_STATS') and config.SHOW_REGION_STATS:
                for region_idx, region_name in enumerate(anatomical_regions):
                    if region_idx in self.region_features_db:
                        count = len(self.region_texts[region_idx])
                        print(f"    - {region_name}: {count} 个样本")
            
        except Exception as e:
            print(f"❌ 加载知识库时出错: {e}")
            self.enabled = False
            return
    
    def retrieve_similar_visual_features(self, query_visual_features, query_image_ids=None, top_k=1):
        """
        基于视觉特征检索相似的视觉特征及其对应的文本描述 - GPU优化版本
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
        
        # 归一化查询特征 - 一次性操作
        query_features_norm = F.normalize(query_visual_features, p=2, dim=2)  # [B, 29, 768]
        
        # 预分配结果张量
        all_similarities = torch.zeros((batch_size, num_regions), device=self.device)
        
        # 优化：预构建image_id到索引的映射，避免重复查找
        image_id_maps = {}
        for region_idx in range(num_regions):
            if region_idx in self.region_features_db:
                region_image_ids = self.region_image_ids[region_idx]
                image_id_maps[region_idx] = {img_id: i for i, img_id in enumerate(region_image_ids)}
        
        # 批量处理：将所有区域的特征拼接，进行一次大规模矩阵乘法
        if hasattr(self, '_combined_features_cache'):
            # 使用缓存的组合特征
            combined_features = self._combined_features_cache
            region_offsets = self._region_offsets_cache
            region_sizes = self._region_sizes_cache
        else:
            # 首次计算时构建缓存
            all_region_features = []
            region_offsets = {}
            region_sizes = {}
            current_offset = 0
            
            for region_idx in range(num_regions):
                if region_idx in self.region_features_db:
                    region_features = self.region_features_db[region_idx]
                    all_region_features.append(region_features)
                    region_offsets[region_idx] = current_offset
                    region_sizes[region_idx] = region_features.size(0)
                    current_offset += region_features.size(0)
                else:
                    region_sizes[region_idx] = 0
            
            if all_region_features:
                combined_features = torch.cat(all_region_features, dim=0)  # [Total_N, 768]
                
                # 缓存结果以备后续使用
                self._combined_features_cache = combined_features
                self._region_offsets_cache = region_offsets
                self._region_sizes_cache = region_sizes
            else:
                combined_features = torch.empty((0, hidden_size), device=self.device)
                region_offsets = {}
                region_sizes = {}
        
        # 如果没有特征数据，返回空结果
        if combined_features.size(0) == 0:
            retrieved_texts = [[""] * num_regions for _ in range(batch_size)]
            return retrieved_texts, all_similarities
        
        # 批量文本结果预分配
        retrieved_texts = []
        
        # 按batch处理，但使用高效的张量操作
        for batch_idx in range(batch_size):
            batch_texts = []
            query_image_id = query_image_ids[batch_idx] 
            
            # 提取当前batch的查询特征 [29, 768]
            batch_query_features = query_features_norm[batch_idx]  
            
            for region_idx in range(num_regions):
                if region_idx not in self.region_features_db or region_sizes[region_idx] == 0:
                    batch_texts.append("")
                    continue
                
                # 获取区域相关数据
                region_offset = region_offsets[region_idx]
                region_size = region_sizes[region_idx] 
                region_texts = self.region_texts[region_idx]
                
                # 提取当前区域的查询特征和数据库特征
                query_feat = batch_query_features[region_idx:region_idx+1]  # [1, 768]
                region_features = combined_features[region_offset:region_offset+region_size]  # [N_region, 768]
                
                # 计算相似度
                similarities = torch.mm(query_feat, region_features.t()).squeeze(0)  # [N_region]
                
                # 处理排除自身的逻辑
                if query_image_id is not None and region_idx in image_id_maps:
                    image_id_map = image_id_maps[region_idx]
                    if query_image_id in image_id_map:
                        exclude_idx = image_id_map[query_image_id]
                        similarities[exclude_idx] = -float('inf')
                
                # 检查是否有有效候选
                valid_similarities = similarities[similarities != -float('inf')]
                if len(valid_similarities) == 0:
                    batch_texts.append("")
                    continue
                
                # 获取top-k结果
                actual_k = min(top_k, len(valid_similarities))
                top_similarities, top_indices = torch.topk(similarities, actual_k)
                
                # 过滤掉无效的结果
                valid_mask = top_similarities != -float('inf')
                if not valid_mask.any():
                    batch_texts.append("")
                    continue
                
                valid_similarities = top_similarities[valid_mask]
                valid_indices = top_indices[valid_mask]
                
                # 记录最高相似度
                all_similarities[batch_idx, region_idx] = valid_similarities[0].item()
                
                # 获取文本描述
                if len(valid_indices) == 1:
                    # 单个结果
                    best_idx = valid_indices[0].item()
                    best_text = region_texts[best_idx]
                    batch_texts.append(best_text)
                else:
                    # 多个结果合并
                    retrieved_descriptions = []
                    indices_cpu = valid_indices.cpu().numpy()
                    
                    for idx in indices_cpu:
                        if 0 <= idx < len(region_texts):
                            retrieved_descriptions.append(region_texts[idx])
                    
                    if retrieved_descriptions:
                        combined_text = " . ".join(retrieved_descriptions)
                        batch_texts.append(combined_text)
                    else:
                        batch_texts.append("")
            
            retrieved_texts.append(batch_texts)
        
        return retrieved_texts, all_similarities
    
    def create_enhanced_prompt(self, source, enhanced_texts, top_sentences=5):
        """创建增强的prompt，使用批量并行处理提高效率
        
        Args:
            source: 包含history等字段的源数据
            enhanced_texts: 每个样本的增强文本列表 [batch_size, list_of_strings]  
            top_sentences: 选择的top句子数量
            
        Returns:
            dict: 增强后的source数据
        """
        # 检查输入的有效性
        if not enhanced_texts or not any(enhanced_texts):
            return source
            
        # 检查history是否为None或无效
        history = source.get("history")
        if history is None:
            return source
            
        # 处理不同类型的history对象
        from transformers.tokenization_utils_base import BatchEncoding
        
        if isinstance(history, BatchEncoding):
            if not hasattr(history, 'input_ids') or history.input_ids is None:
                return source
            history_input_ids = history.input_ids
            history_attention_mask = history.attention_mask
        elif isinstance(history, dict):
            if "input_ids" not in history or history["input_ids"] is None:
                return source
            history_input_ids = history["input_ids"]
            history_attention_mask = history["attention_mask"]
        else:
            return source
            
        device = history_input_ids.device
        batch_size = history_input_ids.size(0)
        original_max_length = history_input_ids.size(1)
        
        # 第一步：批量收集和处理所有增强文本
        batch_enhanced_texts = []
        sample_has_enhancement = []
        
        for i in range(batch_size):
            sample_enhanced_texts = enhanced_texts[i] if enhanced_texts[i] else []
            
            if sample_enhanced_texts:
                # 全局选择top-N句子
                enhanced_scores = []
                for text_score_pair in sample_enhanced_texts:
                    if isinstance(text_score_pair, tuple) and len(text_score_pair) == 2:
                        text, score = text_score_pair
                        enhanced_scores.append((text.strip(), float(score)))
                
                if enhanced_scores:
                    # 按置信度排序并选择top-N
                    enhanced_scores.sort(key=lambda x: x[1], reverse=True)
                    selected_texts = [text for text, _ in enhanced_scores[:top_sentences]]
                    combined_enhanced_text = " ".join(selected_texts)
                    batch_enhanced_texts.append(combined_enhanced_text)
                    sample_has_enhancement.append(True)
                else:
                    batch_enhanced_texts.append("")
                    sample_has_enhancement.append(False)
            else:
                batch_enhanced_texts.append("")
                sample_has_enhancement.append(False)
        
        # 第二步：批量tokenization（只对非空文本进行）
        non_empty_texts = [text for text in batch_enhanced_texts if text.strip()]
        
        if non_empty_texts:
            # 批量编码所有增强文本
            enhanced_encodings = self.tokenizer(
                non_empty_texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,  # 批量padding
                truncation=True,
                max_length=100  # 限制增强文本长度
            ).to(device)
            
            # 批量编码分隔符
            separator_text = " . "
            separator_encoding = self.tokenizer(
                separator_text,
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            separator_ids = separator_encoding["input_ids"].squeeze(0).to(device)  # [sep_len]
        
        # 第三步：批量处理token拼接和padding
        enhanced_input_ids_list = []
        enhanced_attention_mask_list = []
        
        # 为了并行处理，预先计算所有样本的实际长度
        actual_lengths = history_attention_mask.sum(dim=1)  # [batch_size]
        
        enhanced_text_idx = 0
        for i in range(batch_size):
            # 获取当前样本的原始history tokens
            actual_length = actual_lengths[i].item()
            actual_history_ids = history_input_ids[i, :actual_length]  # [actual_len]
            
            if sample_has_enhancement[i] and batch_enhanced_texts[i].strip():
                # 有增强文本的样本
                enhanced_ids = enhanced_encodings.input_ids[enhanced_text_idx]  # [enhanced_len]
                enhanced_mask = enhanced_encodings.attention_mask[enhanced_text_idx]  # [enhanced_len]
                
                # 移除padding（只取有效部分）
                enhanced_actual_length = enhanced_mask.sum().item()
                enhanced_ids = enhanced_ids[:enhanced_actual_length]
                
                # 拼接：original_history + separator + enhanced_text
                combined_ids = torch.cat([actual_history_ids, separator_ids, enhanced_ids])
                enhanced_text_idx += 1
            else:
                # 没有增强文本的样本，保持原始
                combined_ids = actual_history_ids
            
            # 限制最大长度并padding
            max_allowed_length = original_max_length + 50  # 允许适当扩展
            if combined_ids.size(0) > max_allowed_length:
                combined_ids = combined_ids[:max_allowed_length]
            
            # 创建attention mask
            combined_attention_mask = torch.ones(combined_ids.size(0), dtype=torch.long, device=device)
            
            # Padding到统一长度
            if combined_ids.size(0) < max_allowed_length:
                pad_length = max_allowed_length - combined_ids.size(0)
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                
                combined_ids = torch.cat([
                    combined_ids,
                    torch.full((pad_length,), pad_token_id, dtype=torch.long, device=device)
                ])
                combined_attention_mask = torch.cat([
                    combined_attention_mask,
                    torch.zeros(pad_length, dtype=torch.long, device=device)
                ])
            
            enhanced_input_ids_list.append(combined_ids)
            enhanced_attention_mask_list.append(combined_attention_mask)
        
        # 第四步：批量转换为tensor
        enhanced_input_ids = torch.stack(enhanced_input_ids_list)  # [B, max_length]
        enhanced_attention_mask = torch.stack(enhanced_attention_mask_list)  # [B, max_length]
        
        # 更新source中的history
        enhanced_source = source.copy()
        
        # 根据原始history的类型创建相应的返回格式
        if isinstance(history, BatchEncoding):
            enhanced_source["history"] = BatchEncoding({
                "input_ids": enhanced_input_ids,
                "attention_mask": enhanced_attention_mask
            })
        else:
            enhanced_source["history"] = {
                "input_ids": enhanced_input_ids,
                "attention_mask": enhanced_attention_mask
            }
        
        return enhanced_source
    
    def forward(self, visual_features, history_text=None, query_image_ids=None, similarity_threshold=0.5, top_k=1, top_sentences=5):
        """
        前向传播：检索相似的视觉特征并生成增强文本
        
        Args:
            visual_features: 视觉特征张量 [batch_size, num_regions, feature_dim]
            history_text: 历史文本（可选，为了向后兼容）
            query_image_ids: 查询图像ID列表
            similarity_threshold: 相似度阈值
            top_k: 每个区域返回的top相似样本数
            top_sentences: 全局选择的top句子数
            
        Returns:
            enhanced_texts: 每个样本的(文本, 分数)元组列表 [batch_size, list_of_tuples]
        """
        if not self.enabled:
            return None
            
        # 检索相似的视觉特征和对应的文本
        retrieved_texts, similarity_scores = self.retrieve_similar_visual_features(
            visual_features, 
            query_image_ids,
            top_k=top_k
        )
        
        # 转换为(文本, 分数)元组列表格式
        enhanced_texts = []
        batch_size = len(retrieved_texts) if retrieved_texts else 0
        
        if batch_size == 0:
            return None
            
        # 将similarity_scores转换为numpy数组以便处理
        if isinstance(similarity_scores, torch.Tensor):
            similarity_scores_np = similarity_scores.cpu().numpy()
        else:
            similarity_scores_np = np.array(similarity_scores)
        
        for batch_idx in range(batch_size):
            sample_texts = retrieved_texts[batch_idx]
            sample_scores = similarity_scores_np[batch_idx]
            
            # 收集当前样本所有区域的(文本, 分数)对
            sample_enhanced_texts = []
            
            for region_idx, text in enumerate(sample_texts):
                region_score = sample_scores[region_idx]
                
                # 只添加超过阈值的文本
                if region_score >= similarity_threshold and text and text.strip():
                    sample_enhanced_texts.append((text.strip(), region_score))
            
            enhanced_texts.append(sample_enhanced_texts)
        
        return enhanced_texts