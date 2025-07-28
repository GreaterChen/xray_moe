#!/usr/bin/env python3
"""
CSV文件合并脚本
通过image_path作为匹配键拼接两个CSV文件
findings_pred会被重命名为baseline_findings_pred和radmoe_findings_pred来区分
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def merge_csv_files(file1_path, file2_path, output_path, prefix1='baseline', prefix2='radmoe'):
    """
    合并两个CSV文件
    
    Args:
        file1_path (str): 第一个CSV文件路径
        file2_path (str): 第二个CSV文件路径  
        output_path (str): 输出文件路径
        prefix1 (str): 第一个文件findings_pred的前缀，默认为'baseline'
        prefix2 (str): 第二个文件findings_pred的前缀，默认为'radmoe'
    """
    
    try:
        # 读取两个CSV文件
        print(f"正在读取文件1: {file1_path}")
        df1 = pd.read_csv(file1_path)
        
        print(f"正在读取文件2: {file2_path}")
        df2 = pd.read_csv(file2_path)
        
        # 检查必要的列是否存在
        required_columns = ['image_path', 'findings_pred', 'findings_gt', 'labels']
        for col in required_columns:
            if col not in df1.columns:
                raise ValueError(f"文件1缺少必要的列: {col}")
            if col not in df2.columns:
                raise ValueError(f"文件2缺少必要的列: {col}")
        
        print(f"文件1包含 {len(df1)} 行数据")
        print(f"文件2包含 {len(df2)} 行数据")
        
        # 重命名findings_pred列以区分两个文件
        df1_renamed = df1.copy()
        df2_renamed = df2.copy()
        
        df1_renamed = df1_renamed.rename(columns={'findings_pred': f'{prefix1}_findings_pred'})
        df2_renamed = df2_renamed.rename(columns={'findings_pred': f'{prefix2}_findings_pred'})
        
        # 准备合并的列（排除timestamp）
        merge_columns = ['image_path', 'findings_gt', 'labels']
        
        # 验证findings_gt和labels在两个文件中是否一致
        df1_check = df1_renamed[merge_columns + [f'{prefix1}_findings_pred']]
        df2_check = df2_renamed[merge_columns + [f'{prefix2}_findings_pred']]
        
        # 基于image_path进行内连接合并
        merged_df = df1_check.merge(
            df2_check, 
            on='image_path', 
            how='inner',
            suffixes=('_file1', '_file2')
        )
        
        print(f"成功匹配 {len(merged_df)} 行数据")
        
        # 检查findings_gt和labels是否一致
        gt_mismatch = merged_df['findings_gt_file1'] != merged_df['findings_gt_file2']
        labels_mismatch = merged_df['labels_file1'] != merged_df['labels_file2']
        
        if gt_mismatch.any():
            print(f"警告: 发现 {gt_mismatch.sum()} 行的findings_gt不一致")
        
        if labels_mismatch.any():
            print(f"警告: 发现 {labels_mismatch.sum()} 行的labels不一致")
        
        # 创建最终的DataFrame
        final_df = pd.DataFrame({
            'image_path': merged_df['image_path'],
            'findings_gt': merged_df['findings_gt_file1'],  # 使用第一个文件的findings_gt
            f'{prefix1}_findings_pred': merged_df[f'{prefix1}_findings_pred'],
            f'{prefix2}_findings_pred': merged_df[f'{prefix2}_findings_pred'],
            'labels': merged_df['labels_file1']  # 使用第一个文件的labels
        })
        
        # 保存合并后的CSV文件
        print(f"正在保存到: {output_path}")
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"合并完成！")
        print(f"输出文件包含 {len(final_df)} 行数据")
        print(f"列: {list(final_df.columns)}")
        
        return final_df
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='合并两个CSV文件')
    parser.add_argument('--file1', help='第一个CSV文件路径', default="/mnt/chenlb/xray_moe/results/finetune_bert_vit/test_results/val_results_epoch_29.csv")
    parser.add_argument('--file2', help='第二个CSV文件路径', default="/mnt/chenlb/xray_moe/results/finetune_bert_vit_instruction_moe_odd/test_results/val_results_epoch_29.csv") 
    parser.add_argument('-o', '--output', default='merged_results.csv', help='输出文件路径 (默认: merged_results.csv)')
    parser.add_argument('--prefix1', default='baseline', help='第一个文件findings_pred的前缀 (默认: baseline)')
    parser.add_argument('--prefix2', default='radmoe', help='第二个文件findings_pred的前缀 (默认: radmoe)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.file1).exists():
        print(f"错误: 文件不存在: {args.file1}")
        sys.exit(1)
        
    if not Path(args.file2).exists():
        print(f"错误: 文件不存在: {args.file2}")
        sys.exit(1)
    
    # 执行合并
    merge_csv_files(args.file1, args.file2, args.output, args.prefix1, args.prefix2)

if __name__ == "__main__":
    main() 