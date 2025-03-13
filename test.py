import torch
import time

# 设置随机种子以保证结果可重复
torch.manual_seed(42)

# 创建一个14维的查询向量
query_vector = torch.randn(14)

# 创建1636个14维的向量
database_vectors = torch.randn(1636, 14)

# 计算余弦相似度的函数
def cosine_similarity(query, database):
    # 计算查询向量的范数
    query_norm = torch.norm(query)
    
    # 计算数据库中所有向量的范数
    database_norm = torch.norm(database, dim=1)
    
    # 计算点积
    dot_product = torch.matmul(database, query)
    
    # 计算余弦相似度
    similarity = dot_product / (database_norm * query_norm)
    
    return similarity

# 使用PyTorch内置的余弦相似度函数
def cosine_similarity_pytorch(query, database):
    # 调整查询向量的维度以便于计算
    query = query.unsqueeze(0)
    
    # 使用PyTorch的F.cosine_similarity函数计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(database, query, dim=1)
    
    return similarity

# 开始计时
start_time = time.time()

# 计算余弦相似度 - 方法1：自定义实现
similarities = cosine_similarity(query_vector, database_vectors)

# 排序相似度，获取排序后的索引
sorted_indices = torch.argsort(similarities, descending=True)

# 获取排序后的相似度值
sorted_similarities = similarities[sorted_indices]

# 计算运行时间
custom_time = time.time() - start_time
print(f"自定义余弦相似度计算时间: {custom_time:.6f} 秒")

# 重新开始计时
start_time = time.time()

# 计算余弦相似度 - 方法2：使用PyTorch内置函数
similarities_pytorch = cosine_similarity_pytorch(query_vector, database_vectors)

# 排序相似度
sorted_indices_pytorch = torch.argsort(similarities_pytorch, descending=True)
sorted_similarities_pytorch = similarities_pytorch[sorted_indices_pytorch]

# 计算运行时间
pytorch_time = time.time() - start_time
print(f"PyTorch内置函数计算时间: {pytorch_time:.6f} 秒")

# 输出前10个最相似的向量索引及其相似度值
print("\n前10个最相似的向量:")
for i in range(10):
    idx = sorted_indices[i].item()
    sim = sorted_similarities[i].item()
    print(f"索引: {idx}, 相似度: {sim:.4f}")

# 验证两种方法的结果是否相同
is_same = torch.allclose(sorted_similarities, sorted_similarities_pytorch)
print(f"\n两种方法的结果是否相同: {is_same}")