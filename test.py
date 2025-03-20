import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def analyze_trace_file(trace_file_path):
    """分析PyTorch profiler生成的trace.json文件，找出耗时模块"""
    
    # 读取JSON文件
    with open(trace_file_path) as f:
        trace_data = json.load(f)
    
    # 提取events数据
    events = trace_data['traceEvents']
    
    # 初始化一个字典来存储各个模块的耗时
    module_durations = defaultdict(float)
    module_counts = defaultdict(int)
    module_data = []
    
    # 处理每个事件
    for event in events:
        # 只分析完整的事件（有持续时间的事件）
        if 'dur' in event and 'name' in event and 'ph' in event and event['ph'] == 'X':
            name = event['name']
            duration = event['dur'] / 1000  # 转换为毫秒
            
            # 获取事件类别(CPU/CUDA)
            category = event.get('cat', 'unknown')
            
            # 提取模块名称 - 这里我们做一些处理来获取有意义的模块名
            module_name = extract_module_name(name)
            
            # 累加该模块的总耗时
            module_durations[module_name] += duration
            module_counts[module_name] += 1
            
            # 收集详细数据
            module_data.append({
                'module': module_name,
                'name': name,
                'duration_ms': duration,
                'category': category,
                'timestamp': event.get('ts', 0) / 1000,  # 转换为毫秒
                'tid': event.get('tid', 'unknown'),
                'pid': event.get('pid', 'unknown')
            })
    
    # 转换为DataFrame以便分析
    df = pd.DataFrame(module_data)
    
    # 1. 按模块统计总耗时
    module_summary = df.groupby('module')['duration_ms'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
    module_summary['percentage'] = module_summary['sum'] / module_summary['sum'].sum() * 100
    
    # 2. 区分CPU和CUDA操作
    cat_summary = df.groupby(['module', 'category'])['duration_ms'].sum().unstack().fillna(0)
    
    # 3. 分析顺序执行和并行执行
    # 计算理论总执行时间（顺序执行）
    sequential_time = df['duration_ms'].sum()
    
    # 计算实际执行时间（考虑并行）
    # 由于事件可能重叠，实际总时间是从第一个事件开始到最后一个事件结束
    actual_time = df['timestamp'].max() - df['timestamp'].min() + df.loc[df['timestamp'].idxmax(), 'duration_ms']
    
    # 4. 找出瓶颈操作
    top_bottlenecks = df.sort_values('duration_ms', ascending=False).head(20)
    
    # 5. 分析操作的频率
    operation_frequency = df['name'].value_counts().head(20)
    
    # 6. 查找可能的内存问题
    memory_intensive_ops = df[df['name'].str.contains('allocate|free|to|copy', case=False)].groupby('name')['duration_ms'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    
    # 打印结果
    print("===== PyTorch Profiler Trace 分析 =====")
    print(f"\n总执行时间: {actual_time:.2f} ms")
    print(f"理论顺序执行时间: {sequential_time:.2f} ms")
    print(f"并行效率: {sequential_time/actual_time:.2f}x\n")
    
    print("===== 模块耗时排名 (Top 15) =====")
    print(module_summary.head(15))
    
    print("\n===== 各模块CPU/CUDA时间 =====")
    if not cat_summary.empty and 'cuda' in cat_summary.columns:
        print(cat_summary.head(10))
    
    print("\n===== 单次耗时最长的操作 (Top 10) =====")
    print(top_bottlenecks[['name', 'duration_ms', 'category']].head(10))
    
    print("\n===== 调用最频繁的操作 (Top 10) =====")
    print(operation_frequency.head(10))
    
    if not memory_intensive_ops.empty:
        print("\n===== 可能的内存瓶颈操作 =====")
        print(memory_intensive_ops.head(10))
    
    # 可视化数据
    plot_module_analysis(module_summary, cat_summary, df)
    
    return {
        'module_summary': module_summary,
        'cat_summary': cat_summary,
        'top_bottlenecks': top_bottlenecks,
        'operation_frequency': operation_frequency,
        'memory_intensive_ops': memory_intensive_ops,
        'full_data': df
    }

def extract_module_name(name):
    """提取模块名称，处理各种格式的名称"""
    # 针对PyTorch操作提取主要组件
    if name.startswith('aten::'):
        return name.split('(')[0]
    
    # 针对cuDNN和CUDA操作
    if 'cudnn' in name.lower() or 'cuda' in name.lower():
        # 提取主要的cuda/cudnn组件
        cuda_match = re.search(r'(cuda|cudnn)[\w_]+', name.lower())
        if cuda_match:
            return cuda_match.group(0)
    
    # 针对自定义标记的函数
    custom_markers = ['model_forward', 'data_preparation', 'backward_update', 'loss_computation']
    for marker in custom_markers:
        if marker in name:
            return marker
    
    # 针对模型层
    layer_match = re.search(r'(conv\d+|linear\d*|bn\d+|relu|pool\d*|dropout)', name.lower())
    if layer_match:
        return layer_match.group(0)
    
    # 默认返回原始名称
    return name

def plot_module_analysis(module_summary, cat_summary, df):
    """可视化分析结果"""
    plt.figure(figsize=(12, 20))
    
    # 1. 模块总耗时条形图
    plt.subplot(3, 1, 1)
    top_modules = module_summary.head(15)
    sns.barplot(x=top_modules.index, y='sum', data=top_modules.reset_index())
    plt.title('Top 15 Modules by Total Duration')
    plt.ylabel('Duration (ms)')
    plt.xticks(rotation=45, ha='right')
    
    # 2. CPU vs CUDA 耗时(如果有CUDA数据)
    if not cat_summary.empty and 'cuda' in cat_summary.columns:
        plt.subplot(3, 1, 2)
        top_cat = cat_summary.head(10).reset_index()
        top_cat.plot(kind='bar', x='module', figsize=(12, 6))
        plt.title('CPU vs CUDA Duration by Module')
        plt.ylabel('Duration (ms)')
        plt.xticks(rotation=45, ha='right')
    
    # 3. 操作时间线
    plt.subplot(3, 1, 3)
    timeline_data = df.sort_values('timestamp').head(100)  # 显示前100个事件
    sns.scatterplot(x='timestamp', y='module', size='duration_ms', 
                   hue='category', data=timeline_data)
    plt.title('Operations Timeline (first 100 events)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Module')
    
    plt.tight_layout()
    plt.savefig('profile_analysis.png')
    print("\n分析图表已保存为 'profile_analysis.png'")

def recommend_optimizations(analysis_results):
    """基于分析结果提供优化建议"""
    module_summary = analysis_results['module_summary']
    top_bottlenecks = analysis_results['top_bottlenecks']
    operation_frequency = analysis_results['operation_frequency']
    memory_intensive_ops = analysis_results['memory_intensive_ops']
    
    # 初始化建议列表
    recommendations = []
    
    # 1. 分析最耗时的模块
    top_time_modules = module_summary.head(5)
    recommendations.append("## 最耗时模块优化")
    for module, row in top_time_modules.iterrows():
        recommendations.append(f"- **{module}**: 占总时间的 {row['percentage']:.1f}%, 被调用 {row['count']} 次, 平均耗时 {row['mean']:.2f}ms")
    
    # 2. 分析调用次数异常高的操作
    high_freq_ops = operation_frequency.head(5)
    if high_freq_ops.iloc[0] > 1000:  # 如果有操作被调用超过1000次
        recommendations.append("\n## 高频操作优化")
        recommendations.append("以下操作调用频率过高，应考虑批处理或向量化:")
        for op, count in high_freq_ops.items():
            if count > 1000:
                recommendations.append(f"- **{op}**: 调用 {count} 次")
    
    # 3. 分析内存操作
    if not memory_intensive_ops.empty:
        memory_ops = memory_intensive_ops.head(5)
        recommendations.append("\n## 内存操作优化")
        for op, row in memory_ops.iterrows():
            recommendations.append(f"- **{op}**: 总耗时 {row['sum']:.2f}ms, 被调用 {row['count']} 次")
        
        # 检查是否有大量to/copy操作
        if any('to' in op.lower() or 'copy' in op.lower() for op in memory_ops.index):
            recommendations.append("- **建议**: 减少设备间数据传输，预先将数据移至目标设备，避免频繁的CPU-GPU数据移动")
    
    # 4. 模型架构优化建议
    model_ops = [op for op in top_time_modules.index if 'conv' in op.lower() or 'linear' in op.lower() or 'matmul' in op.lower()]
    if model_ops:
        recommendations.append("\n## 模型架构优化")
        recommendations.append("- 考虑使用 `torch.compile()` 优化模型执行")
        recommendations.append("- 检查是否可以降低模型复杂度或使用更高效的层实现")
    
    # 5. 数据加载优化
    if 'data_preparation' in module_summary.index:
        data_prep = module_summary.loc['data_preparation']
        if data_prep['percentage'] > 10:  # 如果数据准备占比超过10%
            recommendations.append("\n## 数据加载优化")
            recommendations.append(f"- 数据准备占总时间的 {data_prep['percentage']:.1f}%")
            recommendations.append("- 增加DataLoader的num_workers")
            recommendations.append("- 使用pin_memory=True加速数据传输")
            recommendations.append("- 考虑预处理和缓存数据集")
    
    # 6. 总体建议
    recommendations.append("\n## 总体优化方向")
    recommendations.append("1. **减少主机-设备内存传输**: 避免频繁使用`.item()`, `.cpu()`, `.numpy()`等导致同步的操作")
    recommendations.append("2. **优化批处理大小**: 找到最佳的batch_size平衡内存使用和计算效率")
    recommendations.append("3. **使用混合精度训练**: 启用`torch.amp`减少内存使用和加速计算")
    recommendations.append("4. **检查梯度计算**: 使用`torch.no_grad()`避免不必要的梯度计算")
    
    return "\n".join(recommendations)

# 主函数
def main(trace_file_path):
    """主函数，分析trace文件并提供优化建议"""
    analysis_results = analyze_trace_file(trace_file_path)
    print("\n\n===== 优化建议 =====")
    print(recommend_optimizations(analysis_results))
    return analysis_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        trace_file = sys.argv[1]
    else:
        trace_file = "trace.json"  # 默认文件名
    
    main(trace_file)