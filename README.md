# SFT-Qwen3-ommi-for-Audio-Reasoning-Challenge


本项目是Interspeech 2026 音频推理挑战（Audio Reasoning Challenge）的解决方法。我们利用

## 快速开始

- **先决条件**：Python 3.8+（推荐 3.10+），Git，若使用 GPU 请安装对应的 CUDA 驱动和 cuDNN。

- **建议环境（示例）**：

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **安装说明**：请根据你使用的硬件（CPU/GPU）调整 `requirements.txt` 中的 `torch` 版本，例如使用 CUDA 版本对应的 `torch` wheel 或官方安装命令。

- **运行示例**：

  ```bash
  # 查看训练脚本帮助
  python scripts/train.py --help

  # 运行一次快速示例（替换为实际脚本与参数）
  python scripts/train.py --config configs/finetune_cot.yaml --data_dir /path/to/data --output_dir ./checkpoints/debug
  ```

请在 `requirements.txt` 中根据目标硬件填入合适的依赖（仓库内已添加示例依赖）。

## 目录结构（示例）

- `data/`：数据集与预处理脚本
- `src/`：主要代码（模型、训练、评估）
- `configs/`：配置文件
- `scripts/`：辅助脚本（训练、推理、评估）

## 贡献

欢迎贡献。请在提交前创建 issue 讨论主要改动，并遵循项目代码风格与测试规范。

## 许可证

请根据实际需要添加许可证信息。

## 联系方式

如需帮助或协作，请打开 issue 或联系仓库维护者。
# Audio-Reasoner 微调 Qwen3-ommi-thinking-30B（基于 Llamaractory） / Fine-tuning Qwen3-ommi-thinking-30B on Audio-Reasoner (Llamaractory)

**简短说明 (Overview)** ✅
- 本仓库记录了使用 **Llamaractory** 框架，基于 **Audio-Reasoner** 数据集对 **Qwen3-ommi-thinking-30B** 进行 COT（Chain-of-Thought）风格微调的完整流程。该 README 为中英双语版，包含训练、推理、评估、模型卡与复现步骤。

- This repo documents the end-to-end process for fine-tuning **Qwen3-ommi-thinking-30B** with **Chain-of-Thought (CoT)** supervision on the **Audio-Reasoner** dataset using **Llamaractory**. This README is bilingual (ZH/EN).

---

## 目录 / Table of Contents
1. 项目简介 / Project Summary
2. 数据与预处理 / Data & Preprocessing
3. 微调配置与训练示例 / Fine-tuning Config & Example Commands
4. 推理与演示 / Inference & Demo
5. 评估 / Evaluation
6. 模型卡与使用声明 / Model Card & Usage Notes
7. 复现说明 / Reproducibility
8. 常见问题 / FAQ
9. 引用 / Citations

---

## 1. 项目简介 / Project Summary
**中文**：本项目目标是让 Qwen3-ommi-thinking-30B 在 Audio-Reasoner 任务上通过 Chain-of-Thought 风格的数据进行微调，从而提高复杂音频推理问题的逐步推理能力与最终答案准确率。

**English**: Goal is to improve step-by-step reasoning and final answer accuracy of Qwen3-ommi-thinking-30B on Audio-Reasoner by fine-tuning with Chain-of-Thought style supervision.

---

## 2. 数据与预处理 / Data & Preprocessing
**中文**：
- 数据集：Audio-Reasoner（请在 `DATA_DIR` 中放置并保持原始结构）。
- 格式：每条样例包含音频标识、转录/特征（如有）、问题、COT 推理过程（训练时的目标）、以及最终答案。
- 预处理要点：
  - 音频处理：如果模型输入包含音频特征（如 log-mel, wav2vec 特征），请在预处理阶段生成 `.npy` / `.pt` 特征并保存索引表。
  - 文本格式化：将 Chain-of-Thought (逐步推理) 与最终答案明确分隔（示例见下）。
  - 数据切分：训练/验证/测试（例如 80/10/10）。

**English**:
- Dataset: Audio-Reasoner (place raw data under `DATA_DIR`).
- Format: each item includes audio id, transcript/features (if any), question, CoT reasoning (target during SFT), and final answer.
- Preprocessing highlights:
  - Audio features: preprocess audio to fixed features if needed (e.g., log-mel, wav2vec). Save indexing table.
  - Text formatting: separate COT chain and final answer explicitly (see example below).
  - Split: Train/Val/Test (e.g., 80/10/10).

Example text training instance (JSON-like):
```
{
  "id": "xxx",
  "audio_feat": "path/to/feat.npy",
  "question": "给出这段音频中说话人的情绪及理由。",
  "cot": "首先…（逐步推理）…因此结论是…", 
  "answer": "愤怒"
}
```

---

## 3. 微调配置与训练示例 / Fine-tuning Config & Example Commands 🔧
**中文说明要点**：
- 推荐使用混合精度（fp16 或 bf16）、梯度累积和适当的 batch-size/学习率策略。30B 模型通常需要大显存（建议 A100 80GB 或使用 ZeRO/分布式策略 / 8bit 存储）。
- 如果使用 LoRA 或参数高效微调 (PEFT)，可显著降低显存需求并保持训练速度。

**Suggested dependencies (示例)**:
- Python >= 3.10
- torch >= 2.x
- transformers
- accelerate / deepspeed
- bitsandbytes (如用 8-bit)
- llama/llamaractory（你使用的 Llamaractory 版本）

示例训练配置 (YAML 模板)：
```yaml
model:
  base_model: qwen3-ommi-thinking-30b
  dtype: bf16
training:
  batch_size: 1               # per device
  gradient_accumulation_steps: 8
  epochs: 3
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_grad_norm: 1.0
optimizer:
  name: adamw
  betas: [0.9, 0.95]
lora:                       # 如果使用 LoRA
  r: 16
  alpha: 32
  dropout: 0.05
data:
  train_file: data/train.jsonl
  val_file: data/val.jsonl
  tokenizer: path/to/tokenizer
logging:
  logging_steps: 50
  save_steps: 2000

```

示例训练命令（根据实际训练脚本调整）:
```bash
# 单节点多卡（示例）
python train.py --config configs/finetune_cot.yaml --data_dir /path/to/data --output_dir ./checkpoints/finetuned_cot

# 使用 accelerate
accelerate launch --config_file accelerate_config.yaml train.py --config configs/finetune_cot.yaml
```

训练要点：
- 使用 seed 固定化以便复现 (e.g., seed=42)
- 定期保存验证检查点并监控验证集上的最终答案准确度和 COT 质量

---

## 4. 推理与演示 / Inference & Demo ▶️
**格式（Prompt Template）**
- 为了引导模型生成 Chain-of-Thought（逐步推理），提示中显式要求推理过程：例如 `请逐步推理并给出最终答案（Step-by-step, then final answer）`。

示例 Prompt（中文/英文双语示例）:
```
System: 你是一个擅长音频推理的助手，请在回答时先列出详细推理步骤（Chain-of-Thought），然后给出最终答案。
User: 问题：<问题文本>
音频描述：<音频转录或特征简述>
请开始逐步推理并给出最终答案。
```

推荐推理超参数：
- temperature: 0.0 - 0.7 (0.0 用于确定性答案)
- top_p: 0.9
- max_new_tokens: 256-512
- stop_sequences: ["\n\n", "Answer:"]

示例推理命令：
```bash
python infer.py --model ./checkpoints/finetuned_cot --prompt_file examples/prompt.jsonl --temperature 0.2 --out predictions.jsonl
```

Self-consistency（稳健性评估）方法：多次采样生成多条 COT 并对最终答案做多数投票，以提高精度。

---

## 5. 评估 / Evaluation ✅
**自动化指标建议**：
- 最终答案准确率（Accuracy on final answer） — 主指标。
- Chain-of-Thought 质量：可采用 BLEU / ROUGE / BERTScore 与参考 COT 比较，但最终仍建议人工标注若干例子进行质量评估。
- Self-consistency 增益测试：比较多次采样后多数投票的准确率提升。

示例评估脚本（伪命令）：
```bash
python eval.py --pred predictions.jsonl --gold data/test.jsonl --metrics accuracy,bleu,rouge
```

人工评估建议：
- 随机抽样 200 个样本，让人工评估 COT 的正确性（每条标注：正确/部分正确/错误），并报告比例。

---

## 6. 模型卡与使用声明 / Model Card & Usage
**主要信息**：
- 模型：基于 Qwen3-ommi-thinking-30B 微调得到的 COT 强化模型。
- 许可：请在此处补充基模型与训练数据许可信息，确保遵守数据和模型许可。

**限制与风险**：
- 对音频转录噪声敏感，错误的音频特征或转录会引起错误推理。
- Chain-of-Thought 生成可能包含不可靠或虚构的中间步骤，请在关键或高风险场景中采用人工核查。

**免责声明**：
- 请勿将本模型用于临床、法律或其他高风险决策场景，除非经过严格的验证和监管合规性审查。

---

## 7. 复现说明 / Reproducibility 🔁
复现要点：
1. 固定随机种子（seed）并记录模型/代码 commit hash。
2. 列出环境依赖（建议提供 `environment.yml` 或 `requirements.txt`）。
3. 提供训练日志与 checkpoint、配置 YAML 和 tokenizer 信息。

示例环境依赖：
```
python==3.10
torch>=2.1
transformers
accelerate
bitsandbytes
numpy
scipy
librosa
llamaractory==<your_version>
```

---

## 8. 常见问题 / FAQ ❓
Q: 如何在内存/显存受限情况下训练 30B 模型？
A: 使用 LoRA/PEFT、8-bit 优化（bitsandbytes）、Deepspeed ZeRO、或分布式多卡训练来降低显存占用。

Q: 如何对 COT 进行自动化评分？
A: 可结合 BLEU/ROUGE/BERTScore 做近似评估，但 COT 的正确性通常需要人工标注或任务特定规则来判定。

---

## 9. 引用 / Citations 📚
- Audio-Reasoner 数据集（请列出数据集论文/仓库引用）
- Qwen3 模型说明（请列出 Qwen 官方引用）
- Llamaractory（请列出对应的项目引用）

---

## 联系与后续工作 / Contact & Next steps
如果你希望我：
- 将训练/评估脚本放进 `scripts/` 下并添加 CI 流程，或
- 将示例推理 notebook (`demo.ipynb`) 添加到仓库，
请告诉我，我可以继续为你实现这些文件并提交到主分支。🔧

---

**License / 许可**：在本仓库中请补充合适的开源许可（如 MIT/Apache-2.0）并注明依赖的模型与数据集许可。

---

谢谢！如果你同意，我现在可以把一份 `scripts/` 示例训练脚本、示例 `configs/finetune_cot.yaml`、以及 `examples/` 的推理/评估脚本一并添加到仓库。🎯
