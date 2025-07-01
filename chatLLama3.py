from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/data/wsh/models/Meta-Llama-3.1-8B-Instruct-128k"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# 构造对话 prompt（LLaMA-3 的 Instruct 格式）
def build_prompt(user_input):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

# 与模型对话
def chat_once(user_input, max_new_tokens=512):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# 示例对话
while True:
    user_input = input("你：")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = chat_once(user_input)
    print("助手：", response)
