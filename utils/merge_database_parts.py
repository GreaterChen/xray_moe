#!/usr/bin/env python3
"""
数据库分批文件合并工具

这个脚本用于将BUILD_DATABASE阶段生成的多个分批文件合并为一个完整的数据库文件。

用法:
    python examples/merge_database_parts.py --input_dir /path/to/database/parts --output /path/to/merged_database.pkl
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm


def merge_database_parts(input_dir, output_path, format_type="auto"):
    """
    合并多个数据库分批文件
    
    Args:
        input_dir: 包含分批文件的目录
        output_path: 输出文件路径
        format_type: 输出格式 ("pkl", "npz", "auto")
    """
    print(f"开始合并数据库分批文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_path}")
    
    # 自动检测格式
    if format_type == "auto":
        if output_path.endswith('.pkl'):
            format_type = "pkl"
        elif output_path.endswith('.npz'):
            format_type = "npz"
        else:
            raise ValueError("无法从输出文件名确定格式，请指定 --format 参数")
    
    # 查找分批文件
    files_list_path = os.path.join(input_dir, "database_files.txt")
    
    if os.path.exists(files_list_path):
        # 从文件列表读取
        with open(files_list_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        file_paths = [os.path.join(input_dir, fname) for fname in filenames]
        print(f"从 database_files.txt 读取到 {len(file_paths)} 个文件")
    else:
        # 自动搜索
        npz_files = sorted([f for f in os.listdir(input_dir) 
                          if f.startswith('anatomical_database_part_') and f.endswith('.npz')])
        pkl_files = sorted([f for f in os.listdir(input_dir) 
                          if f.startswith('anatomical_database_part_') and f.endswith('.pkl')])
        
        if npz_files:
            file_paths = [os.path.join(input_dir, f) for f in npz_files]
            input_format = "npz"
        elif pkl_files:
            file_paths = [os.path.join(input_dir, f) for f in pkl_files]
            input_format = "pkl"
        else:
            raise ValueError("在目录中未找到数据库分批文件")
        
        print(f"自动发现 {len(file_paths)} 个 {input_format.upper()} 文件")
    
    # 检测输入格式
    if file_paths[0].endswith('.npz'):
        input_format = "npz"
    elif file_paths[0].endswith('.pkl'):
        input_format = "pkl"
    else:
        raise ValueError("无法确定输入文件格式")
    
    # 合并数据
    merged_database = {}
    total_images = 0
    total_regions = 0
    
    print(f"\n开始合并 {len(file_paths)} 个文件...")
    
    for i, file_path in enumerate(tqdm(file_paths, desc="合并文件")):
        # 加载文件
        if input_format == "npz":
            npz_data = np.load(file_path, allow_pickle=True)
            part_data = npz_data['image_database'].item()
        elif input_format == "pkl":
            with open(file_path, 'rb') as f:
                part_data = pickle.load(f)
        
        # 统计信息
        part_images = len(part_data)
        part_regions = sum(img_data['detected_count'] for img_data in part_data.values())
        
        print(f"文件 {i+1}: {part_images} 个图像, {part_regions} 个检测区域")
        
        # 检查重复的image_id
        overlapping_keys = set(merged_database.keys()) & set(part_data.keys())
        if overlapping_keys:
            print(f"警告: 发现重复的图像ID: {len(overlapping_keys)} 个")
            print(f"将使用后加载的数据覆盖...")
        
        # 合并数据
        merged_database.update(part_data)
        total_images = len(merged_database)
        total_regions += part_regions
    
    print(f"\n合并完成！")
    print(f"总图像数: {total_images}")
    print(f"总检测区域数: {total_regions}")
    print(f"平均每图像检测区域数: {total_regions / total_images:.2f}")
    
    # 保存合并后的数据库
    print(f"\n保存合并后的数据库到: {output_path}")
    
    if format_type == "npz":
        np.savez_compressed(output_path, image_database=merged_database)
    elif format_type == "pkl":
        with open(output_path, 'wb') as f:
            pickle.dump(merged_database, f)
    
    print(f"合并完成！数据库已保存为 {format_type.upper()} 格式")
    
    return {
        "total_images": total_images,
        "total_regions": total_regions,
        "num_parts_merged": len(file_paths),
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(description="合并数据库分批文件")
    parser.add_argument("--input_dir", help="包含分批文件的目录", default="/mnt/chenlb/datasets/MIMIC/anatomical_database")
    parser.add_argument("--output", help="输出文件路径", default="/mnt/chenlb/datasets/MIMIC/anatomical_database/anatomical_database.npz")
    parser.add_argument("--format", choices=["pkl", "npz", "auto"], default="npz", 
                       help="输出格式 (默认根据文件扩展名自动检测)")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    try:
        result = merge_database_parts(args.input_dir, args.output, args.format)
        print(f"\n✅ 合并成功!")
        print(f"📁 输出文件: {result['output_path']}")
        print(f"📊 图像数量: {result['total_images']}")
        print(f"🎯 检测区域数: {result['total_regions']}")
        print(f"📦 合并文件数: {result['num_parts_merged']}")
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 