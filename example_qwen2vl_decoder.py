#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL Decoder使用示例

展示如何配置和使用Qwen2.5-VL作为decoder进行训练和推理
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import config


def example_config_bert():
    """示例1: 使用BERT decoder (默认配置)"""
    print("=" * 80)
    print("示例1: BERT Decoder配置")
    print("=" * 80)
    
    # 在configs/local_config.py中设置：
    print("""
# configs/local_config.py

PHASE = "FINETUNE_BERT"
DECODER_TYPE = "bert"  # 使用BERT decoder

# 其他配置...
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 5e-5
    """)
    
    print("运行训练:")
    print("  python train.py")
    print()


def example_config_qwen2vl():
    """示例2: 使用Qwen2.5-VL decoder"""
    print("=" * 80)
    print("示例2: Qwen2.5-VL Decoder配置")
    print("=" * 80)
    
    # 在configs/local_config.py中设置：
    print("""
# configs/local_config.py

PHASE = "FINETUNE_BERT"
DECODER_TYPE = "qwen2vl"  # 使用Qwen2.5-VL decoder
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # 或本地路径

# 注意：Qwen2.5-VL需要更多显存，建议调整批次大小
TRAIN_BATCH_SIZE = 16  # 从64减小到16
VAL_BATCH_SIZE = 8     # 从16减小到8

# 其他配置...
EPOCHS = 50
LEARNING_RATE = 5e-5
    """)
    
    print("运行训练:")
    print("  python train.py")
    print()


def example_local_qwen_model():
    """示例3: 使用本地Qwen2.5-VL模型"""
    print("=" * 80)
    print("示例3: 使用本地Qwen2.5-VL模型")
    print("=" * 80)
    
    print("""
如果你已经下载了Qwen2.5-VL模型到本地，可以这样配置：

# configs/local_config.py

DECODER_TYPE = "qwen2vl"
QWEN_MODEL_NAME = "/path/to/local/Qwen2.5-VL-3B-Instruct"  # 本地路径

# 下载模型到本地的方法：
# 1. 使用huggingface-cli:
#    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /path/to/local/Qwen2.5-VL-3B-Instruct
#
# 2. 或使用git:
#    git lfs install
#    git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct /path/to/local/Qwen2.5-VL-3B-Instruct
    """)
    print()


def example_inference():
    """示例4: 推理时使用Qwen2.5-VL"""
    print("=" * 80)
    print("示例4: 推理时使用Qwen2.5-VL")
    print("=" * 80)
    
    print("""
# configs/local_config.py

PHASE = "INFER_BERT"  # 推理阶段
DECODER_TYPE = "qwen2vl"  # 使用Qwen2.5-VL
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# 生成参数可调整
GEN_DO_SAMPLE = True
GEN_NUM_BEAMS = 3
GEN_MAX_NEW_TOKENS = 150
GEN_MIN_LENGTH = 100
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.0
    """)
    
    print("运行推理:")
    print("  python compare_inference_strategies.py")
    print()


def example_comparison():
    """示例5: 对比BERT和Qwen2.5-VL"""
    print("=" * 80)
    print("示例5: 对比BERT和Qwen2.5-VL性能")
    print("=" * 80)
    
    print("""
你可以分别训练两个模型，然后对比它们的性能：

步骤1: 训练BERT decoder
---------------------------------------
# configs/local_config.py
DECODER_TYPE = "bert"
CHECKPOINT_PATH_TO = "/path/to/checkpoints/bert_decoder"

运行: python train.py

步骤2: 训练Qwen2.5-VL decoder
---------------------------------------
# configs/local_config.py
DECODER_TYPE = "qwen2vl"
CHECKPOINT_PATH_TO = "/path/to/checkpoints/qwen2vl_decoder"

运行: python train.py

步骤3: 对比推理结果
---------------------------------------
分别加载两个checkpoint进行推理，对比生成质量：
- BLEU, ROUGE, METEOR等指标
- CheXbert临床指标
- 人工评估生成的报告质量
    """)
    print()


def show_feature_flow():
    """示例6: 特征处理流程说明"""
    print("=" * 80)
    print("示例6: Qwen2.5-VL的特征处理流程")
    print("=" * 80)
    
    print("""
特征处理流程：

1. 视觉特征提取
   输入: 医学影像
   输出: [B, 30, 768]  (1个CLS + 29个解剖区域)
   
2. 疾病特征提取（如果启用RGAT）
   输入: 解剖区域特征
   输出: [B, 14, 768]  (14个疾病类别)

3. 特征映射
   视觉特征: [B, 30, 768] -> visual_projection -> [B, 30, qwen_hidden_dim]
   疾病特征: [B, 14, 768] -> disease_projection -> [B, 14, qwen_hidden_dim]
   
4. 特征拼接
   组合特征 = [视觉特征, 疾病特征]
   输出: [B, 44, qwen_hidden_dim]

5. 文本生成
   输入: 组合特征 [B, 44, qwen_hidden_dim]
   输出: 生成的医学报告文本

关键点：
- 视觉和疾病特征有独立的映射层
- 映射层是可训练的，适应Qwen2.5-VL的表示空间
- Qwen2.5-VL模型可以冻结或微调
    """)
    print()


def show_memory_optimization():
    """示例7: 显存优化建议"""
    print("=" * 80)
    print("示例7: 显存优化建议")
    print("=" * 80)
    
    print("""
Qwen2.5-VL (3B参数) 需要较多显存，以下是优化建议：

1. 减小批次大小
   TRAIN_BATCH_SIZE = 8   # 从64减小
   VAL_BATCH_SIZE = 4     # 从16减小

2. 使用梯度累积
   # 在trainer中添加
   accumulation_steps = 8  # 等效批次大小 = 8 * 8 = 64
   
3. 冻结Qwen模型（只训练映射层）
   # 在qwen2vl_decoder.py中添加：
   for param in self.qwen_model.parameters():
       param.requires_grad = False
       
4. 使用混合精度训练
   USE_MIXED_PRECISION = True  # 已默认启用
   
5. 使用梯度检查点（Gradient Checkpointing）
   self.qwen_model.gradient_checkpointing_enable()

显存需求估算：
- 24GB (RTX 3090/4090): 批次大小 8-16
- 40GB (A100): 批次大小 16-32
- 80GB (A100 80GB): 批次大小 32-64
    """)
    print()


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Qwen2.5-VL Decoder 使用示例                                ║
╚════════════════════════════════════════════════════════════════════════════╝

本脚本展示如何配置和使用Qwen2.5-VL作为decoder。

当前配置:
  - DECODER_TYPE: {decoder_type}
  - PHASE: {phase}
  - QWEN_MODEL_NAME: {qwen_model}
    """.format(
        decoder_type=getattr(config, 'DECODER_TYPE', 'bert'),
        phase=getattr(config, 'PHASE', 'UNK'),
        qwen_model=getattr(config, 'QWEN_MODEL_NAME', 'N/A')
    ))
    
    # 显示所有示例
    example_config_bert()
    example_config_qwen2vl()
    example_local_qwen_model()
    example_inference()
    example_comparison()
    show_feature_flow()
    show_memory_optimization()
    
    print("=" * 80)
    print("更多详细信息，请参考:")
    print("  docs/QWEN2VL_DECODER_USAGE.md")
    print("=" * 80)


if __name__ == "__main__":
    main()


