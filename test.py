import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# --- 模型和分词器加载 ---
model_path = "/home/chenlb/.cache/modelscope/hub/models/prithivMLmods/Llama-Doctor-3.2-3B-Instruct"
print(f"Loading model from: {model_path}")
model = LlamaForCausalLM.from_pretrained(model_path)
print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Llama 通常没有专门的 pad token，会用 eos_token 代替
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# --- 准备输入 ---
prompt = "Hey! Who are you?"
print(f"\nPrompt: '{prompt}'")

# 1. Tokenize (获取 input_ids 和 attention_mask)
inputs = tokenizer(prompt, return_tensors="pt", padding=True) # 使用 padding 以防万一
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
print(f"Input IDs: {input_ids.tolist()}")
print(f"Attention Mask: {attention_mask.tolist()}")

# 2. 获取 Embedding 层
embedding_layer = model.get_input_embeddings()
print(f"Embedding layer: {embedding_layer}")

# 3. 将 input_ids 转换为 embeds (模拟你代码中的做法)
# 注意：确保 input_ids 在正确的设备上 (如果模型在 GPU 上)
# model.device 会告诉你模型在哪个设备
input_ids = input_ids.to(model.device)
prompt_embeds = embedding_layer(input_ids)
print(f"Shape of prompt_embeds: {prompt_embeds.shape}")

# --- 执行生成 (使用 inputs_embeds) ---
print("\n--- Generating using inputs_embeds ---")
# 当使用 inputs_embeds 时，通常也需要提供 attention_mask
# 使用与之前测试相同的 max_length
generate_ids_embeds = model.generate(
    inputs_embeds=prompt_embeds,
    attention_mask=attention_mask.to(model.device), # 确保 mask 也在同一设备
    max_length=30,
    pad_token_id=tokenizer.pad_token_id # 指定 pad_token_id 是好习惯
)
print(f"Generated IDs (from embeds): {generate_ids_embeds.tolist()}")

# --- 解码输出 ---
print("\n--- Decoding output from inputs_embeds ---")

# 解码 (不跳过特殊 token)
text_embeds_special = tokenizer.batch_decode(generate_ids_embeds, skip_special_tokens=False)[0]
print(f"\nDecoded (skip_special_tokens=False):\n'{text_embeds_special}'")

# 解码 (跳过特殊 token)
text_embeds_no_special = tokenizer.batch_decode(generate_ids_embeds, skip_special_tokens=True)[0]
print(f"\nDecoded (skip_special_tokens=True):\n'{text_embeds_no_special}'")


# --- 对比：使用 input_ids 生成 (你之前的测试) ---
print("\n\n--- For Comparison: Generating using input_ids ---")
generate_ids_ids = model.generate(
    input_ids=input_ids, # 直接使用 IDs
    attention_mask=attention_mask.to(model.device),
    max_length=30,
    pad_token_id=tokenizer.pad_token_id
)
print(f"Generated IDs (from input_ids): {generate_ids_ids.tolist()}")
text_ids_compare = tokenizer.batch_decode(generate_ids_ids, skip_special_tokens=True)[0]
print(f"\nDecoded from input_ids (skip_special_tokens=True):\n'{text_ids_compare}'")