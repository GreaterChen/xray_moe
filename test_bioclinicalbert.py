"""
测试BioClinicalBERT配置是否正确
"""

import torch
from transformers import AutoTokenizer, BertConfig, BertLMHeadModel

def test_bioclinicalbert():
    print("=" * 50)
    print("测试BioClinicalBERT配置")
    print("=" * 50)
    
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    # 1. 测试tokenizer加载
    print("\n1. 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer加载成功")
        print(f"   词表大小: {len(tokenizer)}")
        print(f"   特殊token: CLS={tokenizer.cls_token}, SEP={tokenizer.sep_token}, PAD={tokenizer.pad_token}")
        
        # 添加自定义特殊token
        special_tokens = {}
        if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
            special_tokens["bos_token"] = "[DEC]"
        if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
            special_tokens["eos_token"] = "[EOS]"
        
        if special_tokens:
            num_added = tokenizer.add_special_tokens(special_tokens)
            print(f"   添加了 {num_added} 个特殊token")
            print(f"   新词表大小: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return False
    
    # 2. 测试配置加载
    print("\n2. 加载模型配置...")
    try:
        config = BertConfig.from_pretrained(model_name)
        print(f"✅ 配置加载成功")
        print(f"   原始vocab_size: {config.vocab_size}")
        
        # 设置交叉注意力
        config.add_cross_attention = True
        config.is_decoder = True
        config.encoder_width = 768
        print(f"   已启用交叉注意力")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 3. 测试模型加载
    print("\n3. 加载预训练模型...")
    try:
        model = BertLMHeadModel.from_pretrained(model_name, config=config)
        print(f"✅ 模型加载成功")
        print(f"   模型vocab_size: {model.config.vocab_size}")
        
        # 调整词表大小以匹配tokenizer
        model.resize_token_embeddings(len(tokenizer))
        print(f"   调整后vocab_size: {model.config.vocab_size}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 4. 测试前向传播
    print("\n4. 测试前向传播...")
    try:
        # 创建测试输入
        test_text = "The chest x-ray shows clear lungs with no evidence of consolidation."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        # 创建dummy encoder hidden states (模拟视觉特征)
        batch_size = inputs.input_ids.shape[0]
        encoder_hidden_states = torch.randn(batch_size, 30, 768)  # 30个视觉token
        encoder_attention_mask = torch.ones(batch_size, 30, dtype=torch.long)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=inputs.input_ids  # 用于计算loss
            )
        
        print(f"✅ 前向传播成功")
        print(f"   Loss: {outputs.loss.item():.4f}")
        print(f"   Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！BioClinicalBERT配置正确")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_bioclinicalbert()
    if not success:
        print("\n❌ 测试失败，请检查配置")
        exit(1)
