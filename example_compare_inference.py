#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理策略对比示例代码
展示如何在Python代码中使用InferenceComparator类
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import config
from compare_inference_strategies import InferenceComparator


def example_full_comparison():
    """示例1: 完整对比（使用默认的5种策略）"""
    print("=" * 80)
    print("示例1: 完整对比实验")
    print("=" * 80)
    
    comparator = InferenceComparator(
        config=config,
        checkpoint_path="/path/to/your/checkpoint.pth",  # 修改为实际路径
        device="cuda",
        output_dir="./comparison_results"
    )
    
    # 运行完整对比（使用全部测试集）
    comparator.run_comparison()


def example_quick_test():
    """示例2: 快速测试（使用子集）"""
    print("=" * 80)
    print("示例2: 快速测试（100个样本）")
    print("=" * 80)
    
    comparator = InferenceComparator(
        config=config,
        checkpoint_path="/path/to/your/checkpoint.pth",  # 修改为实际路径
        device="cuda",
        output_dir="./quick_test_results"
    )
    
    # 运行快速测试（只用100个样本）
    comparator.run_comparison(subset_size=100)


def example_custom_strategies():
    """示例3: 自定义策略对比"""
    print("=" * 80)
    print("示例3: 自定义策略对比")
    print("=" * 80)
    
    comparator = InferenceComparator(
        config=config,
        checkpoint_path="/path/to/your/checkpoint.pth",  # 修改为实际路径
        device="cuda",
        output_dir="./custom_comparison"
    )
    
    # 创建测试数据加载器
    test_loader = comparator.create_test_loader(subset_size=50)
    
    # 定义自定义策略
    custom_strategies = {
        "my_beam_search": {
            "num_beams": 4,
            "do_sample": False,
            "max_new_tokens": 150,
            "min_length": 80,
            "repetition_penalty": 1.2
        },
        "my_sampling": {
            "num_beams": 1,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 150,
            "min_length": 80,
            "repetition_penalty": 1.1
        }
    }
    
    all_results = []
    all_metrics = []
    
    # 对每个策略进行测试
    for strategy_name, params in custom_strategies.items():
        # 生成
        results = comparator.generate_with_strategy(
            test_loader, 
            strategy_name, 
            params
        )
        results['params'] = params
        all_results.append(results)
        
        # 评估
        metrics = comparator.evaluate_results(results, strategy_name)
        metrics['params'] = params
        all_metrics.append(metrics)
    
    # 保存结果
    comparator.save_results(all_results, all_metrics)


def example_single_strategy():
    """示例4: 测试单一策略"""
    print("=" * 80)
    print("示例4: 测试单一策略")
    print("=" * 80)
    
    comparator = InferenceComparator(
        config=config,
        checkpoint_path="/path/to/your/checkpoint.pth",  # 修改为实际路径
        device="cuda",
        output_dir="./single_strategy_test"
    )
    
    # 创建测试数据加载器
    test_loader = comparator.create_test_loader(subset_size=20)
    
    # 定义单一策略
    strategy_params = {
        "num_beams": 3,
        "do_sample": False,
        "max_new_tokens": 150,
        "min_length": 100,
        "repetition_penalty": 1.0
    }
    
    # 生成
    results = comparator.generate_with_strategy(
        test_loader, 
        "test_strategy", 
        strategy_params
    )
    
    # 评估
    metrics = comparator.evaluate_results(results, "test_strategy")
    
    # 查看一些生成示例
    print("\n生成示例:")
    print("=" * 80)
    for i in range(min(3, len(results['generated']))):
        print(f"\n样本 {i+1}:")
        print(f"生成: {results['generated'][i][:200]}...")
        print(f"真实: {results['ground_truth'][i][:200]}...")


if __name__ == "__main__":
    print("""
请选择要运行的示例:
1. 完整对比实验（5种策略，全部测试集）
2. 快速测试（5种策略，100个样本）
3. 自定义策略对比（2种自定义策略，50个样本）
4. 单一策略测试（1种策略，20个样本）

注意: 运行前请修改代码中的 checkpoint_path 为实际的模型权重路径！
    """)
    
    choice = input("请输入选项 (1-4): ").strip()
    
    try:
        if choice == "1":
            example_full_comparison()
        elif choice == "2":
            example_quick_test()
        elif choice == "3":
            example_custom_strategies()
        elif choice == "4":
            example_single_strategy()
        else:
            print("无效的选项！")
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保修改代码中的 checkpoint_path 为实际的模型权重路径！")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

