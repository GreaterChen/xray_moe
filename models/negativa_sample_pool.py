import numpy as np
import torch
import torch.nn.functional as F


class NegativeSamplePool:
    """
    负样本池管理类：按疾病标签组合组织报告的CLS token，
    支持基于余弦相似度选择负样本
    """
    def __init__(self, num_diseases=14, similarity_threshold=0.99):
        """
        初始化负样本池
        
        Args:
            num_diseases: 疾病类别数量
        """
        self.pool = {}  # 键为标签向量，值为对应的token tensor列表
        self.label_vectors = {}  # 键为字符串键，值为原始标签tensor
        self.num_diseases = num_diseases
        self.similarity_threshold = similarity_threshold
        
        # 记录添加的样本数和去重的样本数
        self.sample_count = 0
        self.duplicate_count = 0
        
        # 添加缓存
        self.similarity_cache = {}  # 缓存标签相似度计算结果
        self.cache_size = 2000  # 最大缓存条目数
        
        # 检查是否可以使用CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def _label_to_key(self, label):
        """
        将标签向量转换为字符串键
        
        Args:
            label: 疾病标签向量 [num_diseases]
            
        Returns:
            key: 标签向量的字符串表示
        """
        # 将标签向量(浮点数)转换为0/1的布尔字符串
        bool_list = [1 if val > 0.5 else 0 for val in label.cpu().numpy()]
        return str(bool_list)
    
    def _is_token_duplicate(self, token, token_tensor):
        """
        检查token是否在token_tensor中已存在（基于余弦相似度）
        
        Args:
            token: 要检查的token向量 (tensor)
            token_tensor: 已有token的tensor [num_tokens, hidden_dim]
            
        Returns:
            is_duplicate: 是否存在相似度超过阈值的token
        """
        if token_tensor.size(0) == 0:
            return False
            
        # 确保token是适当的形状 [1, hidden_dim]
        token = token.unsqueeze(0)
        
        # 计算余弦相似度 [num_tokens]
        similarities = torch.nn.functional.cosine_similarity(token_tensor, token, dim=1)
        
        # 检查是否有相似度超过阈值的token
        return torch.any(similarities > self.similarity_threshold).item()
    
    def update(self, text_cls_tokens, labels):
        """
        更新负样本池，并进行去重
        
        Args:
            text_cls_tokens: 报告的CLS token [batch_size, hidden_dim]
            labels: 疾病标签 [batch_size, num_diseases]
        """
        batch_size = labels.shape[0]
        batch_duplicates = 0
        
        for i in range(batch_size):
            # 获取当前样本的标签和token
            sample_label = labels[i]  # [num_diseases]
            sample_token = text_cls_tokens[i].to(self.device)  # [hidden_dim]
            
            # 将标签向量转换为键
            label_key = self._label_to_key(sample_label)
            
            # 存储原始标签向量（用于后续计算相似度）
            if label_key not in self.label_vectors:
                self.label_vectors[label_key] = sample_label.to(self.device)
            
            # 如果该组合不存在，创建一个新tensor
            if label_key not in self.pool:
                self.pool[label_key] = torch.empty((0, sample_token.size(0)), device=self.device)
            
            # 检查是否已存在相同token
            if not self._is_token_duplicate(sample_token, self.pool[label_key]):
                # 添加新token到对应的标签组合池中
                self.pool[label_key] = torch.cat([self.pool[label_key], sample_token.unsqueeze(0)], dim=0)
            else:
                # 记录重复token
                batch_duplicates += 1
                self.duplicate_count += 1
            
            # 更新样本计数（包括重复样本）
            self.sample_count += 1
            
            # 每添加1000个样本打印一次进度
            if self.sample_count % 1000 == 0:
                duplicate_percentage = (self.duplicate_count / self.sample_count) * 100
                print(f"已处理 {self.sample_count} 个样本，去重 {self.duplicate_count} 个 ({duplicate_percentage:.2f}%)")
        
        # 每批次结束打印本批次的去重信息
        if batch_duplicates > 0:
            print(f"当前批次: 处理 {batch_size} 个样本，去重 {batch_duplicates} 个")
    
    def save(self, save_path):
        """
        保存负样本池到本地
        
        Args:
            save_path: 保存路径
        """
        print("保存负样本池到本地...")
        
        # 处理池数据并保存
        save_data = {
            'pool': {},
            'label_vectors': {}
        }
        
        # 将tensor转换为numpy数组
        for label_key, tokens in self.pool.items():
            save_data['pool'][label_key] = tokens.cpu().numpy()
        
        for label_key, label_vector in self.label_vectors.items():
            save_data['label_vectors'][label_key] = label_vector.cpu().numpy()
        
        # 使用numpy保存
        np.save(save_path, save_data)
        
        print(f"负样本池已保存至 {save_path}")
        print("池统计信息:")
        total_samples = 0
        for label_key, tokens in save_data['pool'].items():
            num_samples = tokens.shape[0]
            print(f"标签组合 {label_key}: {num_samples} 个样本")
            total_samples += num_samples
        print(f"总计: {total_samples} 个样本")
    
    def load(self, load_path):
        """
        从本地加载负样本池，并转换为tensor放在GPU上
        
        Args:
            load_path: 加载路径
        """
        print(f"从 {load_path} 加载负样本池...")
        
        # 加载数据
        loaded_data = np.load(load_path, allow_pickle=True).item()
        
        # 将numpy数组转换为tensor并放在GPU上
        self.pool = {}
        self.label_vectors = {}
        
        # 统计信息收集
        total_samples = 0
        key_sizes = []
        
        # 将pool数据转为tensor
        for label_key, tokens_array in loaded_data['pool'].items():
            if tokens_array.size > 0:  # 确保非空
                tokens_tensor = torch.tensor(tokens_array, dtype=torch.float32, device=self.device)
                self.pool[label_key] = tokens_tensor
                num_samples = tokens_tensor.size(0)
            else:
                # 创建空tensor
                self.pool[label_key] = torch.empty((0, tokens_array.shape[1] if tokens_array.shape else 0), 
                                                   device=self.device)
                num_samples = 0
                
            total_samples += num_samples
            key_sizes.append((label_key, num_samples))
        
        # 将label_vectors数据转为tensor
        for label_key, label_array in loaded_data['label_vectors'].items():
            self.label_vectors[label_key] = torch.tensor(label_array, dtype=torch.float32, device=self.device)
        
        print(f"总计: {total_samples} 个样本")
        print(f"总键数: {len(self.pool)}")
        
        # 按样本数量排序
        key_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 输出前五个最大的键
        print("\n样本数量最多的前五个标签组合:")
        for i, (key, size) in enumerate(key_sizes[:5], 1):
            print(f"{i}. 标签组合 {key}: {size} 个样本")
        
        # 输出后五个最小的键
        print("\n样本数量最少的前五个标签组合:")
        for i, (key, size) in enumerate(key_sizes[-5:], 1):
            print(f"{i}. 标签组合 {key}: {size} 个样本")
    
    def _compute_all_similarities(self, target_label):
        """
        计算目标标签与所有存储标签的相似度，使用缓存加速
        """
        # 将target_label转换为字符串键
        target_key = self._label_to_key(target_label)
        
        # 检查缓存
        if target_key in self.similarity_cache:
            return self.similarity_cache[target_key]
            
        # 计算相似度
        target_tensor = target_label.to(self.device)
        keys = list(self.label_vectors.keys())
        similarities_list = []
        
        if keys:
            # 批量计算相似度
            vectors_tensor = torch.stack([self.label_vectors[key] for key in keys], dim=0)
            similarities = F.cosine_similarity(vectors_tensor, target_tensor.unsqueeze(0), dim=1)
            
            # 创建并排序结果
            similarities_list = [(keys[i], similarities[i].item()) for i in range(len(keys))]
            similarities_list.sort(key=lambda x: x[1])
            
            # 更新缓存
            if len(self.similarity_cache) >= self.cache_size:
                # 移除最旧的缓存条目
                oldest_key = next(iter(self.similarity_cache))
                del self.similarity_cache[oldest_key]
            
            self.similarity_cache[target_key] = similarities_list
        
        return similarities_list
    
    def get_negative_samples_batch(self, target_labels, k=10):
        """
        批量获取多个目标标签的负样本 - 优化版
        
        Args:
            target_labels: 目标标签向量批次 [batch_size, num_diseases]
            k: 每个目标标签需要的负样本数量
                
        Returns:
            negative_samples_batch: 包含每个目标的负样本的列表
        """
        batch_size = target_labels.shape[0]
        
        # 1. 批量计算所有目标标签与所有存储标签的相似度
        batch_similarities = []
        target_keys = []
        
        # 为每个目标标签生成键并获取相似度列表
        for i in range(batch_size):
            target_label = target_labels[i]
            target_key = self._label_to_key(target_label)
            target_keys.append(target_key)
            
            # 检查缓存或计算相似度
            if target_key in self.similarity_cache:
                batch_similarities.append(self.similarity_cache[target_key])
            else:
                similarities = self._compute_all_similarities(target_label)
                batch_similarities.append(similarities)
        
        # 2. 批量收集负样本
        negative_samples_batch = []
        
        # 预先计算每个label_key的样本数量和样本总量
        available_samples = {}
        for label_key, samples in self.pool.items():
            available_samples[label_key] = samples.size(0)
        
        # 并行处理每个目标标签
        for i in range(batch_size):
            sorted_similarities = batch_similarities[i]
            
            # 使用torch.cat一次性收集样本
            collected_samples = []
            total_samples = 0
            
            # 对于相似度排序的每个标签组合
            for label_key, _ in sorted_similarities:
                current_size = available_samples.get(label_key, 0)
                if current_size == 0:
                    continue
                    
                samples_needed = k - total_samples
                if samples_needed <= 0:
                    break
                    
                # 获取当前标签组合对应的池
                current_samples = self.pool[label_key]
                
                # 选择适当数量的样本
                if current_size <= samples_needed:
                    collected_samples.append(current_samples)
                    total_samples += current_size
                else:
                    # 随机选择部分样本
                    selected_indices = torch.randperm(current_size, device=self.device)[:samples_needed]
                    selected_samples = current_samples[selected_indices]
                    collected_samples.append(selected_samples)
                    total_samples += samples_needed
            
            # 合并所有收集到的样本
            if collected_samples:
                all_negative = torch.cat(collected_samples, dim=0)
                negative_samples = all_negative[:k] if all_negative.size(0) > k else all_negative
            else:
                negative_samples = None
                
            negative_samples_batch.append(negative_samples)
        
        return negative_samples_batch