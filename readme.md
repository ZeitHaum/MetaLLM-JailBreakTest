-----

# LLM 黑盒越狱攻击评估工具

-----

本项目提供了一套用于评估大型语言模型（LLMs）在**黑盒越狱攻击**下安全鲁棒性的工具。它集成了多种经典的提示工程越狱策略，并支持对 Llama 系列模型进行批量测试，旨在帮助研究人员和开发者深入理解模型的安全边界及其对恶意输入的抵抗能力。

## 项目概览

本项目专注于 LLM 的**黑盒越狱攻击**评估。通过模拟真实世界中攻击者可能采用的各种提示工程技巧，我们测试目标模型（默认为 Llama 3.1 Instruct）在面对恶意指令时的响应，并量化其越狱成功率。这有助于揭示模型在不同攻击场景下的脆弱点，为模型安全对齐的改进提供数据支持。

-----

## 特性

  * **多种越狱策略：** 内置多种主流黑盒越狱攻击方法，覆盖了从简单伪装到复杂多轮交互的场景。
  * **批量自动化测试：** 能够加载批量有害指令，对模型进行自动化评估。
  * **结果记录：** 详细记录每次测试的原始提示、模型响应及越狱成功状态。
  * **成功率统计：** 自动统计每种攻击策略的越狱成功率。
  * **易于扩展：** 模块化的代码结构方便添加新的越狱策略或集成其他模型。
  * **模型兼容性：** 默认支持 Llama 3 和 Llama 2 的聊天模板，易于扩展到其他 Hugging Face 兼容模型。

-----

## 攻击策略

本项目实现了以下黑盒越狱攻击策略：

1.  **Vanilla (原版)：** 不做任何修改的原始有害指令。
2.  **Role Playing (角色扮演)：** 诱导模型扮演一个没有限制的虚拟角色，以绕过其安全约束。
3.  **Few-Shot Jailbreak (少样本越狱)：** 通过提供无害的问答示例，试图混淆模型，使其在后续有害指令上放松警惕。
4.  **Self-Instruct Jailbreak (自指令越狱)：** 模拟用户分步引导模型生成有害内容的过程。
5.  **AutoDAN：** 采用自动化攻击的思路，通过引导模型进入“无限制”模式。
6.  **Unicode Obfuscation (Unicode 混淆)：** 在敏感词中插入零宽度字符等 Unicode 干扰字符，试图绕过关键词过滤。
7.  **Multi-Q Adv (多问题对抗)：** 在有害指令前插入多个无关问题，试图分散模型注意力。
8.  **JSON Wrapper (JSON 封装)：** 将有害指令封装在 JSON 结构中，伪装成特定格式的数据。
9.  **Prompt Leakage Injection (提示泄漏注入)：** 伪造系统提示泄露，试图诱导模型遵循虚假指令。

-----

## 环境准备

1.  **克隆仓库：**

    ```bash
    git clone <your_repo_url>
    cd <your_repo_name>
    ```

2.  **创建 Conda 环境 (推荐)：**

    ```bash
    conda create -n llm_jailbreak python=3.10
    conda activate llm_jailbreak
    ```

3.  **安装依赖：**

    ```bash
    pip install torch transformers pandas numpy accelerate
    ```

      * **注意：** 请根据你的 GPU 型号和 PyTorch 版本选择合适的 `torch` 安装命令。例如，CUDA 12.1 用户可参考 PyTorch 官网安装。
      * `accelerate` 是为了更好地管理 GPU 内存和推理。

4.  **下载模型：**
    本项目默认使用 Llama 3.1 模型。你需要从 Hugging Face Hub 下载模型到本地路径。确保你有访问 Meta Llama 模型的权限。
    默认模型路径：`/data/wsh/models/Meta-Llama-3.1-8B-Instruct-128k`。请根据你的实际情况修改 `main.py` 或通过命令行参数 `--model_path` 指定。

-----

## 使用方法

通过 `run_eval.sh` 脚本执行评估。你可以自定义模型路径和测试样本数量。

**运行命令：**

```bash
sh run_eval.sh
```

-----

## 输出结果

程序运行完成后，会在项目根目录下创建一个 `results` 文件夹，其中包含两个 CSV 文件：

  * `results/prompts.csv`: 记录了每次测试所使用的详细提示信息，包括 `sample_id` (样本ID), `attack_type` (攻击类型), `prompt` (发送给模型的完整提示)。
  * `results/responses.csv`: 记录了模型对每次攻击的响应，包括 `sample_id`, `attack_type`, `response` (模型的原始回复)。

控制台还会输出每个攻击策略的最终越狱成功率（Attack Success Rate）。

-----

## 代码结构

  * `main.py`: 主程序入口，负责参数解析、模型加载、测试用例加载、攻击执行和结果统计。
  * `eval_utils.py`: 包含 `is_refusal_response` 函数，用于判断模型响应是否为安全拒绝（即越狱失败）。
  * `run_eval.sh`: 简化运行命令的 Bash 脚本。
  * `advbench/harmful_behaviors.csv`: 存储有害指令的 CSV 文件。

-----

## 注意事项

  * **GPU 内存：** 大型语言模型需要大量 GPU 内存。请确保你的显卡具备足够的 VRAM。
  * **模型访问权限：** 某些 Llama 模型可能需要申请 Hugging Face 的访问权限。请确保你的 Hugging Face 账户已获得相应权限，并在本地配置好认证信息（例如，通过 `huggingface-cli login`）。
  * **越狱定义：** `eval_utils.is_refusal_response` 函数的逻辑直接影响越狱成功率的计算。如有需要，请根据你的评估标准调整该函数的实现。
  * **合规性：** 本项目旨在学术研究和安全评估，请勿将代码用于任何非法或不道德的活动。

-----