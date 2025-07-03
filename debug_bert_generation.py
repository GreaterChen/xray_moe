import torch
import os
import sys
from transformers import BertTokenizer

# 添加项目路径
sys.path.append('/home/chenlb/xray_moe')

from configs import config
from models.bert_cross_decoder import BertCrossDecoder
from models.moe_bert_adapter import MoEBertAdapter

def test_bert_generation():
    print("=== 测试BERT生成功能 ===")
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
    # 创建BERT解码器
    bert_adapter = MoEBertAdapter(
        config=config,
        tokenizer=tokenizer,
        hidden_dim=768,
        max_length=100
    )
    
    bert_adapter = bert_adapter.cuda()
    bert_adapter.eval()
    
    # 创建测试数据
    batch_size = 2
    num_tokens = 30  # 1 CLS + 29 regions
    hidden_dim = 768
    
    # 模拟视觉特征
    visual_features = torch.randn(batch_size, num_tokens, hidden_dim).cuda()
    
    print(f"视觉特征形状: {visual_features.shape}")
    
    # 测试1: 没有history的生成
    print("\n--- 测试1: 无history生成 ---")
    try:
        generated_texts_no_history = bert_adapter.generate(
            visual_features=visual_features,
            history_encoding=None,
            use_history=False,
            max_new_tokens=20,
            do_sample=False,  # 使用greedy decoding
            num_beams=1
        )
        print(f"无history生成成功: {len(generated_texts_no_history)} 个样本")
        for i, text in enumerate(generated_texts_no_history):
            print(f"  样本{i}: '{text}'")
    except Exception as e:
        print(f"无history生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 有history的生成（空history）
    print("\n--- 测试2: 空history生成 ---")
    try:
        empty_history = ["", ""]  # 空字符串列表
        generated_texts_empty_history = bert_adapter.generate(
            visual_features=visual_features,
            history_encoding=empty_history,
            use_history=True,
            max_new_tokens=20,
            do_sample=False,  # 使用greedy decoding
            num_beams=1
        )
        print(f"空history生成成功: {len(generated_texts_empty_history)} 个样本")
        for i, text in enumerate(generated_texts_empty_history):
            print(f"  样本{i}: '{text}'")
    except Exception as e:
        print(f"空history生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bert_generation()
