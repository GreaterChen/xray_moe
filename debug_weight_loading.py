#!/usr/bin/env python3
"""
诊断权重加载问题：对比FINETUNE_BERT和INFER_BERT阶段的权重加载差异
"""

import torch
import torch.nn as nn
import os
import sys
from transformers import BertTokenizer

# 添加项目路径
sys.path.append('/home/chenlb/xray_moe')

from models.moe_model import MOE
from models.fast_rcnn_classifier import DetectionOnlyFastRCNN, EnhancedFastRCNN
from models.vit import MedicalVisionTransformer
from models.bert_cross_decoder import BertCrossDecoder
from models.moe_bert_adapter import MoEBertAdapter
from utils import load, count_parameters
from configs import config

def compare_weight_loading():
    """对比两种权重加载方式的效果"""
    print("=== 诊断权重加载问题 ===")
    
    checkpoint_path = '/mnt/chenlb/xray_moe/results/finetune_bert_vit_instruction_moe_odd_extra_2itc/epoch_49_bleu_0.6189_ce_f1_0.5746.pth'
    
    # 1. 模拟FINETUNE_BERT阶段的权重加载方式
    print("\n--- 模拟FINETUNE_BERT阶段权重加载 ---")
    
    # 创建模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    
    detection_model = DetectionOnlyFastRCNN()
    enhanced_rcnn = EnhancedFastRCNN(
        pretrained_detector=detection_model, num_regions=29, feature_dim=768
    )
    vit_model = MedicalVisionTransformer()
    bert_model = MoEBertAdapter(
        config=config,
        tokenizer=tokenizer,
        hidden_dim=768,
        max_length=100
    )
    
    model_finetune = MOE(
        config=config,
        object_detector=enhanced_rcnn,
        image_encoder=vit_model,
        findings_decoder=bert_model,
    )
    
    # FINETUNE_BERT方式：将权重加载到model.findings_decoder.decoder
    print("FINETUNE_BERT加载方式：加载到 model.findings_decoder.decoder")
    _, _ = load(checkpoint_path, model_finetune.findings_decoder.decoder, load_model="decoder")
    
    # 提取BERT关键权重作为参考
    finetune_weights = {}
    bert_decoder = model_finetune.findings_decoder.decoder.text_decoder.bert
    if hasattr(bert_decoder.embeddings.word_embeddings, 'weight'):
        finetune_weights['word_embeddings'] = bert_decoder.embeddings.word_embeddings.weight.clone()
    if hasattr(bert_decoder.encoder.layer[0].attention.self.query, 'weight'):
        finetune_weights['first_layer_query'] = bert_decoder.encoder.layer[0].attention.self.query.weight.clone()
    
    print(f"FINETUNE方式加载后的word embeddings前5个值: {finetune_weights['word_embeddings'][0, :5]}")
    print(f"FINETUNE方式加载后的第一层query权重前5个值: {finetune_weights['first_layer_query'][0, :5]}")
    
    # 2. 模拟INFER_BERT阶段的权重加载方式  
    print("\n--- 模拟INFER_BERT阶段权重加载 ---")
    
    # 重新创建相同的模型（以确保权重被重置）
    tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    tokenizer2.add_special_tokens({'bos_token': '[DEC]'})
    
    detection_model2 = DetectionOnlyFastRCNN()
    enhanced_rcnn2 = EnhancedFastRCNN(
        pretrained_detector=detection_model2, num_regions=29, feature_dim=768
    )
    vit_model2 = MedicalVisionTransformer()
    bert_model2 = MoEBertAdapter(
        config=config,
        tokenizer=tokenizer2,
        hidden_dim=768,
        max_length=100
    )
    
    model_infer = MOE(
        config=config,
        object_detector=enhanced_rcnn2,
        image_encoder=vit_model2,
        findings_decoder=bert_model2,
    )
    
    # INFER_BERT方式：将权重加载到整个model
    print("INFER_BERT加载方式：加载到整个 model")
    _, _ = load(checkpoint_path, model_infer, load_model="full")
    
    # 提取BERT关键权重进行对比
    infer_weights = {}
    bert_decoder_infer = model_infer.findings_decoder.decoder.text_decoder.bert
    if hasattr(bert_decoder_infer.embeddings.word_embeddings, 'weight'):
        infer_weights['word_embeddings'] = bert_decoder_infer.embeddings.word_embeddings.weight.clone()
    if hasattr(bert_decoder_infer.encoder.layer[0].attention.self.query, 'weight'):
        infer_weights['first_layer_query'] = bert_decoder_infer.encoder.layer[0].attention.self.query.weight.clone()
    
    print(f"INFER方式加载后的word embeddings前5个值: {infer_weights['word_embeddings'][0, :5]}")
    print(f"INFER方式加载后的第一层query权重前5个值: {infer_weights['first_layer_query'][0, :5]}")
    
    # 3. 对比权重是否一致
    print("\n--- 权重对比结果 ---")
    
    # 对比word embeddings
    embeddings_equal = torch.allclose(finetune_weights['word_embeddings'], infer_weights['word_embeddings'], atol=1e-6)
    print(f"Word embeddings是否一致: {embeddings_equal}")
    
    if not embeddings_equal:
        diff = torch.abs(finetune_weights['word_embeddings'] - infer_weights['word_embeddings'])
        max_diff = torch.max(diff)
        print(f"Word embeddings最大差异: {max_diff}")
        print(f"Word embeddings平均差异: {torch.mean(diff)}")
    
    # 对比第一层query权重
    query_equal = torch.allclose(finetune_weights['first_layer_query'], infer_weights['first_layer_query'], atol=1e-6)
    print(f"第一层query权重是否一致: {query_equal}")
    
    if not query_equal:
        diff = torch.abs(finetune_weights['first_layer_query'] - infer_weights['first_layer_query'])
        max_diff = torch.max(diff)
        print(f"第一层query权重最大差异: {max_diff}")
        print(f"第一层query权重平均差异: {torch.mean(diff)}")
    
    # 4. 检查权重加载的详细信息
    print("\n--- 检查权重加载路径匹配 ---")
    
    # 读取检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 检查BERT相关的关键权重是否存在
    key_weights = [
        'findings_decoder.decoder.text_decoder.bert.embeddings.word_embeddings.weight',
        'findings_decoder.decoder.text_decoder.bert.encoder.layer.0.attention.self.query.weight',
        'findings_decoder.decoder.text_decoder.cls.predictions.decoder.weight',
        'findings_decoder.decoder.text_decoder.cls.predictions.bias'
    ]
    
    for key in key_weights:
        if key in state_dict:
            print(f"✅ 检查点中存在: {key}")
            print(f"   形状: {state_dict[key].shape}")
            print(f"   前5个值: {state_dict[key].view(-1)[:5]}")
        else:
            print(f"❌ 检查点中缺失: {key}")
    
    # 5. 检查模型中对应权重的路径
    print("\n--- 检查模型中的权重路径 ---")
    model_state = model_infer.state_dict()
    
    for key in key_weights:
        if key in model_state:
            print(f"✅ 模型中存在: {key}")
            print(f"   形状: {model_state[key].shape}")
            print(f"   前5个值: {model_state[key].view(-1)[:5]}")
        else:
            print(f"❌ 模型中缺失: {key}")
            
    # 6. 总结
    print("\n=== 诊断总结 ===")
    if embeddings_equal and query_equal:
        print("✅ 两种加载方式的权重完全一致，权重加载没有问题")
        print("问题可能出在其他地方，比如模型状态、生成参数等")
    else:
        print("❌ 两种加载方式的权重不一致，这就是问题所在！")
        print("建议检查load函数中的路径匹配逻辑")

if __name__ == "__main__":
    compare_weight_loading() 