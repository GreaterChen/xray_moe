"""
双层异构知识图谱 + 三阶段关系感知图注意力网络 (RGAT)
Relational Graph Attention Network for Medical Diagnosis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from configs.constants import ANATOMY_ORDER, DISEASE_ORDER


class RelationalGraphAttentionLayer(nn.Module):
    """
    关系型图注意力层 - 支持异构节点和边
    """
    def __init__(self, in_dim, out_dim, use_edge_weight=False, dropout=0.1, negative_slope=0.2):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            use_edge_weight: 是否使用边权重(用于DD层,边权重为Jaccard系数)
            dropout: Dropout率
            negative_slope: LeakyReLU的负斜率
        """
        super(RelationalGraphAttentionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_weight = use_edge_weight
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # 特征变换矩阵
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # 注意力权重向量
        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, h, adj_matrix, edge_weights=None):
        """
        Args:
            h: 节点特征 [batch_size, num_nodes, in_dim]
            adj_matrix: 邻接矩阵 [num_nodes, num_nodes] - 0/1二值矩阵或连续权重矩阵
            edge_weights: 可选的边权重 [num_nodes, num_nodes] - 用于DD层的Jaccard系数
            
        Returns:
            h_prime: 更新后的节点特征 [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = h.shape
        device = h.device
        
        # 特征变换: [batch_size, num_nodes, out_dim]
        Wh = self.W(h)
        
        # 计算注意力系数
        # a_input: [batch_size, num_nodes, num_nodes, 2*out_dim]
        # 对于每条边(i,j),拼接 [Wh_i || Wh_j]
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)  # [B, N*N, out_dim]
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)  # [B, N*N, out_dim]
        
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        all_combinations = all_combinations.view(batch_size, num_nodes, num_nodes, 2 * self.out_dim)
        
        # 计算注意力得分: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e = self.leaky_relu(torch.matmul(all_combinations, self.a).squeeze(-1))  # [B, N, N]
        
        # 应用边权重(对于DD层)
        if self.use_edge_weight and edge_weights is not None:
            e = e * edge_weights.unsqueeze(0)  # [B, N, N] * [1, N, N]
        
        # 根据邻接矩阵mask掉不存在的边
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix.unsqueeze(0) > 0, e, zero_vec)  # [B, N, N]
        
        # Softmax归一化(沿着邻居维度)
        attention = F.softmax(attention, dim=2)  # [B, N, N]
        attention = self.dropout(attention)
        
        # 聚合邻居特征
        h_prime = torch.bmm(attention, Wh)  # [B, N, out_dim]
        
        return h_prime


class HeterogeneousGraphAttentionLayer(nn.Module):
    """
    异构图注意力层 - 用于处理不同类型节点之间的交互(A→D)
    源节点和目标节点使用不同的变换矩阵
    """
    def __init__(self, source_dim, target_dim, out_dim, dropout=0.1, negative_slope=0.2):
        """
        Args:
            source_dim: 源节点特征维度(Anatomy节点)
            target_dim: 目标节点特征维度(Disease节点)
            out_dim: 输出特征维度
            dropout: Dropout率
            negative_slope: LeakyReLU的负斜率
        """
        super(HeterogeneousGraphAttentionLayer, self).__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # 源节点和目标节点的特征变换矩阵
        self.W_source = nn.Linear(source_dim, out_dim, bias=False)
        self.W_target = nn.Linear(target_dim, out_dim, bias=False)
        
        # 注意力权重向量
        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W_source.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_target.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, h_source, h_target, adj_matrix):
        """
        从源节点(Anatomy)聚合信息到目标节点(Disease)
        
        Args:
            h_source: 源节点特征 [batch_size, num_source, source_dim]
            h_target: 目标节点特征 [batch_size, num_target, target_dim]
            adj_matrix: 邻接矩阵 [num_source, num_target] - 表示A→D的连接
            
        Returns:
            h_target_prime: 更新后的目标节点特征 [batch_size, num_target, out_dim]
        """
        batch_size = h_source.shape[0]
        num_source = h_source.shape[1]
        num_target = h_target.shape[1]
        device = h_source.device
        
        # 变换源节点和目标节点特征
        Wh_source = self.W_source(h_source)  # [B, num_source, out_dim]
        Wh_target = self.W_target(h_target)  # [B, num_target, out_dim]
        
        # 为每个目标节点d,计算所有源节点a的注意力
        # [B, num_target, num_source, 2*out_dim]
        
        # 扩展维度以便拼接
        # Wh_source: [B, 1, num_source, out_dim] -> [B, num_target, num_source, out_dim]
        Wh_source_expanded = Wh_source.unsqueeze(1).expand(-1, num_target, -1, -1)
        
        # Wh_target: [B, num_target, 1, out_dim] -> [B, num_target, num_source, out_dim]
        Wh_target_expanded = Wh_target.unsqueeze(2).expand(-1, -1, num_source, -1)
        
        # 拼接: [B, num_target, num_source, 2*out_dim]
        concat = torch.cat([Wh_source_expanded, Wh_target_expanded], dim=-1)
        
        # 计算注意力得分: e_ad = LeakyReLU(a^T [Wh_a || Wh_d])
        e = self.leaky_relu(torch.matmul(concat, self.a).squeeze(-1))  # [B, num_target, num_source]
        
        # 根据邻接矩阵mask掉不存在的边 (注意adj_matrix是[num_source, num_target],需要转置)
        adj_matrix_t = adj_matrix.t()  # [num_target, num_source]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix_t.unsqueeze(0) > 0, e, zero_vec)  # [B, num_target, num_source]
        
        # Softmax归一化(沿着源节点维度)
        attention = F.softmax(attention, dim=2)  # [B, num_target, num_source]
        attention = self.dropout(attention)
        
        # 聚合源节点特征到目标节点
        h_target_prime = torch.bmm(attention, Wh_source)  # [B, num_target, out_dim]
        
        return h_target_prime


class DiseaseClassifierHead(nn.Module):
    """
    疾病分类头 - 为每个疾病节点添加二分类器
    """
    def __init__(self, disease_dim, dropout=0.3):
        super(DiseaseClassifierHead, self).__init__()
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(disease_dim, disease_dim // 2),
                nn.LayerNorm(disease_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(disease_dim // 2, 1),
            )
            for _ in range(14)  # 14个疾病
        ])
    
    def forward(self, disease_features):
        """
        Args:
            disease_features: [batch_size, 14, disease_dim]
            
        Returns:
            disease_preds: [batch_size, 14] - 14个疾病的预测结果
        """
        batch_size = disease_features.shape[0]
        device = disease_features.device
        
        preds = []
        for i in range(14):
            pred = self.classifiers[i](disease_features[:, i, :])  # [B, 1]
            preds.append(pred)
        
        disease_preds = torch.cat(preds, dim=1)  # [B, 14]
        return disease_preds


class ThreeStageRGAT(nn.Module):
    """
    三阶段关系感知图注意力网络
    
    阶段1: 解剖区域上下文感知 (A→A)
    阶段2: 疾病特异性表征聚合 (A→D)
    阶段3: 疾病间关系推理 (D→D)
    """
    def __init__(
        self,
        num_anatomy=29,
        num_disease=14,
        anatomy_dim=768,  # ViT输出的区域特征维度
        disease_dim=768,  # 疾病节点嵌入维度
        hidden_dim_1=768,  # 阶段1输出维度
        hidden_dim_2=768,  # 阶段2输出维度
        hidden_dim_3=768,  # 阶段3输出维度
        dropout=0.1,
        aa_adj_path=None,  # A→A邻接矩阵路径
        dd_adj_path=None,  # D→D邻接矩阵路径(带权重)
        da_adj_path=None,  # D→A(实际是A→D)邻接矩阵路径
    ):
        super(ThreeStageRGAT, self).__init__()
        
        self.num_anatomy = num_anatomy
        self.num_disease = num_disease
        
        # 疾病节点的可学习嵌入
        self.disease_embeddings = nn.Parameter(torch.randn(num_disease, disease_dim))
        nn.init.normal_(self.disease_embeddings, mean=0, std=0.02)
        
        # 阶段1: 解剖区域上下文感知 (A→A)
        self.stage1_aa = RelationalGraphAttentionLayer(
            in_dim=anatomy_dim,
            out_dim=hidden_dim_1,
            use_edge_weight=False,
            dropout=dropout
        )
        
        # 阶段2: 疾病特异性表征聚合 (A→D)
        self.stage2_ad = HeterogeneousGraphAttentionLayer(
            source_dim=hidden_dim_1,
            target_dim=disease_dim,
            out_dim=hidden_dim_2,
            dropout=dropout
        )
        
        # 阶段3: 疾病间关系推理 (D→D)
        self.stage3_dd = RelationalGraphAttentionLayer(
            in_dim=hidden_dim_2,
            out_dim=hidden_dim_3,
            use_edge_weight=True,  # DD层使用边权重
            dropout=dropout
        )
        
        # 疾病分类头
        self.disease_classifier = DiseaseClassifierHead(hidden_dim_3, dropout=dropout)
        
        # 加载图结构
        self.register_buffer("aa_adj", self._load_adjacency_matrix(aa_adj_path, (num_anatomy, num_anatomy)))
        self.register_buffer("dd_adj", self._load_adjacency_matrix(dd_adj_path, (num_disease, num_disease)))
        self.register_buffer("dd_weights", self._load_adjacency_matrix(dd_adj_path, (num_disease, num_disease), as_weights=True))
        self.register_buffer("ad_adj", self._load_adjacency_matrix(da_adj_path, (num_anatomy, num_disease)))
        
        print(f"✅ RGAT模块初始化完成")
        print(f"   - 解剖区域节点数: {num_anatomy}")
        print(f"   - 疾病节点数: {num_disease}")
        print(f"   - A→A边数: {self.aa_adj.sum().item():.0f}")
        print(f"   - A→D边数: {self.ad_adj.sum().item():.0f}")
        print(f"   - D→D边数: {self.dd_adj.sum().item():.0f}")
    
    def _load_adjacency_matrix(self, path, shape, as_weights=False):
        """
        加载邻接矩阵
        
        Args:
            path: CSV文件路径
            shape: 矩阵形状 (num_rows, num_cols)
            as_weights: 是否作为权重矩阵(保留原始值),否则转为0/1二值矩阵
            
        Returns:
            adj_matrix: torch.Tensor [num_rows, num_cols]
        """
        if path is None or not isinstance(path, str):
            print(f"⚠️ 邻接矩阵路径无效: {path}, 使用全连接图")
            return torch.ones(shape)
        
        try:
            # 读取CSV文件（第一行是表头，第一列是行名，已按标准顺序排列）
            df = pd.read_csv(path, header=0, index_col=0)
            matrix = df.values.astype(np.float32)
            
            # 验证顺序一致性
            if shape == (29, 29):
                # AA矩阵
                if list(df.columns) != ANATOMY_ORDER or list(df.index) != ANATOMY_ORDER:
                    print(f"⚠️ AA矩阵顺序与代码定义不一致!")
            elif shape == (14, 14):
                # DD矩阵
                if list(df.columns) != DISEASE_ORDER or list(df.index) != DISEASE_ORDER:
                    print(f"⚠️ DD矩阵顺序与代码定义不一致!")
            elif shape == (29, 14):
                # DA矩阵
                if list(df.index) != ANATOMY_ORDER or list(df.columns) != DISEASE_ORDER:
                    print(f"⚠️ DA矩阵顺序与代码定义不一致!")
            
            # 检查形状
            if matrix.shape != shape:
                print(f"⚠️ 邻接矩阵形状不匹配: 期望{shape}, 实际{matrix.shape}")
                # 尝试调整
                if matrix.shape[0] > shape[0]:
                    matrix = matrix[:shape[0], :]
                if matrix.shape[1] > shape[1]:
                    matrix = matrix[:, :shape[1]]
                if matrix.shape[0] < shape[0] or matrix.shape[1] < shape[1]:
                    # 填充
                    new_matrix = np.zeros(shape, dtype=np.float32)
                    new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
                    matrix = new_matrix
            
            # 转换为tensor
            adj_tensor = torch.from_numpy(matrix).float()
            
            # 如果不是权重矩阵,转为0/1二值矩阵
            if not as_weights:
                adj_tensor = (adj_tensor > 0).float()
            
            print(f"✅ 成功加载邻接矩阵: {path.split('/')[-1]}")
            print(f"   - 形状: {adj_tensor.shape}")
            print(f"   - 边数/非零元素: {(adj_tensor > 0).sum().item():.0f}")
            if as_weights:
                print(f"   - 权重范围: [{adj_tensor.min().item():.4f}, {adj_tensor.max().item():.4f}]")
            
            return adj_tensor
            
        except Exception as e:
            print(f"❌ 加载邻接矩阵失败 {path}: {e}")
            import traceback
            traceback.print_exc()
            print(f"   使用全连接图替代")
            return torch.ones(shape)
    
    def forward(self, anatomy_features, return_attention=False):
        """
        三阶段推理
        
        Args:
            anatomy_features: [batch_size, num_anatomy, anatomy_dim] - ViT输出的29个解剖区域特征
            return_attention: 是否返回注意力权重(用于可视化)
            
        Returns:
            disease_features: [batch_size, num_disease, hidden_dim_3] - 14个疾病特征
            disease_preds: [batch_size, num_disease] - 14个疾病的分类预测
        """
        batch_size = anatomy_features.shape[0]
        device = anatomy_features.device
        
        # 扩展疾病嵌入到batch维度
        disease_init = self.disease_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 14, disease_dim]
        
        # === 阶段1: 解剖区域上下文感知 (A→A) ===
        anatomy_contextualized = self.stage1_aa(
            anatomy_features,
            self.aa_adj
        )  # [B, 29, hidden_dim_1]
        anatomy_contextualized = F.elu(anatomy_contextualized)
        
        # === 阶段2: 疾病特异性表征聚合 (A→D) ===
        disease_specific = self.stage2_ad(
            anatomy_contextualized,
            disease_init,
            self.ad_adj
        )  # [B, 14, hidden_dim_2]
        disease_specific = F.elu(disease_specific)
        
        # === 阶段3: 疾病间关系推理 (D→D) ===
        disease_features = self.stage3_dd(
            disease_specific,
            self.dd_adj,
            edge_weights=self.dd_weights
        )  # [B, 14, hidden_dim_3]
        disease_features = F.elu(disease_features)
        
        # === 疾病分类 ===
        disease_preds = self.disease_classifier(disease_features)  # [B, 14]
        
        return disease_features, disease_preds
    
    def compute_classification_loss(self, disease_preds, labels):
        """
        计算疾病分类损失
        
        Args:
            disease_preds: [batch_size, 14] - 疾病预测结果(logits)
            labels: [batch_size, 14] - 疾病标签(0/1)
            
        Returns:
            loss: 二分类交叉熵损失
        """
        # 数值稳定性检查
        if torch.isnan(disease_preds).any() or torch.isinf(disease_preds).any():
            print("⚠️ disease_preds包含NaN或Inf值")
            disease_preds = torch.nan_to_num(disease_preds, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print("⚠️ labels包含NaN或Inf值")
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 限制预测值范围
        disease_preds = torch.clamp(disease_preds, min=-10.0, max=10.0)
        
        # 计算二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(disease_preds, labels)
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ RGAT分类损失出现NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=disease_preds.device, requires_grad=True)
        
        return loss

