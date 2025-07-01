from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
model_path = "/home/wsh/LLMSecurity-LuoWenJian/autodl-tmp/Llama-2-7b-chat-hf"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model.eval()  # 明确设为评估模式，避免训练态影响

# 构建 prompt（适配 LLaMA-2，而不是 LLaMA-3）
def build_prompt(user_input):
    return f"[INST] {user_input} [/INST]"

# 单轮对话函数
@torch.no_grad()  # 禁用梯度计算，加速推理
def chat_once(user_input, max_new_tokens=512):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# 命令行交互
if __name__ == "__main__":
    while True:
        try:
            user_input = input("你：")
            if user_input.lower() in {"exit", "quit"}:
                break
            response = chat_once(user_input)
            print("助手：", response)
        except KeyboardInterrupt:
            print("\n退出。")
            break
