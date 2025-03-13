import numpy as np
import torch


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
        计算目标标签与所有存储标签的相似度
        
        Args:
            target_label: 目标标签向量 [num_diseases]
            
        Returns:
            similarities: 包含(key, similarity)元组的列表，按相似度升序排序
        """
        # 确保目标标签是设备上的tensor
        target_tensor = target_label.to(self.device)
        
        # 创建标签键和向量的列表
        keys = list(self.label_vectors.keys())
        similarities_list = []
        
        if keys:
            # 将所有标签向量堆叠成一个批次
            vectors_tensor = torch.stack([self.label_vectors[key] for key in keys], dim=0)
            
            # 批量计算余弦相似度
            similarities = torch.nn.functional.cosine_similarity(vectors_tensor, target_tensor.unsqueeze(0), dim=1)
            
            # 创建(key, similarity)元组列表
            similarities_list = [(keys[i], similarities[i].item()) for i in range(len(keys))]
            
            # 按相似度升序排序（最不相似的在前面）
            similarities_list.sort(key=lambda x: x[1])
        
        return similarities_list
    
    def get_negative_samples(self, target_label, k=10):
        """
        获取与目标标签相似度最低的k个样本
        
        Args:
            target_label: 目标标签向量 [num_diseases]
            k: 需要的负样本数量
            
        Returns:
            negative_samples: 负样本tensor [k, hidden_dim]
        """
        # 计算目标标签与所有标签的相似度并排序
        sorted_similarities = self._compute_all_similarities(target_label)
        
        # 按相似度升序遍历，收集k个样本
        collected_samples = []
        
        for label_key, similarity in sorted_similarities:
            # 获取当前标签的样本tensor
            current_samples = self.pool[label_key]
            
            # 确保有样本可用
            if current_samples.size(0) == 0:
                continue
                
            # 计算要从当前池中选择的样本数
            samples_needed = k - len(collected_samples)
            
            # 如果当前池中样本数量足够，随机选择需要的数量
            if current_samples.size(0) <= samples_needed:
                collected_samples.append(current_samples)
            else:
                # 随机选择不重复的样本索引
                selected_indices = torch.randperm(current_samples.size(0), device=self.device)[:samples_needed]
                selected_samples = current_samples[selected_indices]
                collected_samples.append(selected_samples)
            
            # 如果已收集足够样本，退出循环
            if sum(tensor.size(0) for tensor in collected_samples) >= k:
                break
        
        # 如果收集到的样本非空，将它们连接成一个tensor
        total_collected = sum(tensor.size(0) for tensor in collected_samples)
        
        if total_collected == 0:
            print("警告: 未找到任何负样本")
            return None
            
        if total_collected < k:
            print(f"警告: 只找到 {total_collected} 个负样本，少于请求的 {k} 个")
        
        # 连接所有收集到的样本
        negative_samples = torch.cat(collected_samples, dim=0)
        
        # 如果样本超过了k个，截取前k个
        if negative_samples.size(0) > k:
            negative_samples = negative_samples[:k]
            
        return negative_samples
    
    def get_negative_samples_batch(self, target_labels, k=10):
        """
        批量获取多个目标标签的负样本
        
        Args:
            target_labels: 目标标签向量批次 [batch_size, num_diseases]
            k: 每个目标标签需要的负样本数量
            
        Returns:
            negative_samples_batch: 包含每个目标的负样本的列表
        """
        batch_size = target_labels.shape[0]
        negative_samples_batch = []
        
        for i in range(batch_size):
            negative_samples = self.get_negative_samples(target_labels[i], k)
            negative_samples_batch.append(negative_samples)
        
        return negative_samples_batch