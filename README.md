# SFT-Qwen3-ommi-for-Audio-Reasoning-Challenge

# Audio-Reasoner Fine-tuning Qwen3-ommi-thinking-30B on Audio-Reasoner (Llamaractory)

**Overview**:
This repo documents the end-to-end process for fine-tuning **Qwen3-ommi-thinking-30B** with **Chain-of-Thought (CoT)** supervision on the **Audio-Reasoner** dataset using **Llamaractory**, including data processing, training, inference, evaluation, and reproducibility.

---

## Table of Contents
1. Project Summary
2. Data & Preprocessing
3. Fine-tuning Config & Example Commands
4. Inference & Demo
5. Evaluation
6. Model Card & Usage Notes
7. Reproducibility
8. FAQ
9. Citations

---

## 1. Project Summary

## 2. Data & Preprocessing
- Dataset: [Audio-Reasoner-CoTA](https://huggingface.co/datasets/zhifeixie/Audio-Reasoner-CoTA) (place raw data under `DATA_DIR`).
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
  "question": "Give the emotion and reasoning of the speaker in this audio.",
  "cot": "First... (step-by-step reasoning)... therefore the conclusion is...", 
  "answer": "anger"
}
```

---

## 3. Fine-tuning Config & Example Commands 🔧
**Suggested dependencies (example)**:
- Python >= 3.10
- torch >= 2.x
- transformers
- accelerate / deepspeed
- bitsandbytes (if using 8-bit)
- llama/llamaractory (the version you are using)

Example training configuration (YAML template):
```yaml
model:
  base_model: qwen3-ommi-thinking-30b
  dtype: bf16
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  epochs: 3
  lr: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_grad_norm: 1.0
optimizer:
  name: adamw
  betas: [0.9, 0.95]
lora:
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

Example training commands (adjust to your script):
```bash
# single-node multi-GPU (example)
python train.py --config configs/finetune_cot.yaml --data_dir /path/to/data --output_dir ./checkpoints/finetuned_cot

# using accelerate
accelerate launch --config_file accelerate_config.yaml train.py --config configs/finetune_cot.yaml
```

Training notes:
- Use mixed precision (fp16 or bf16), gradient accumulation, appropriate batch size/learning rate.
- For 30B model, large memory needed (recommend A100 80GB or ZeRO/distributed, 8bit).
- If using LoRA or PEFT, memory can be reduced while keeping training speed.
- Fix seed for reproducibility (e.g., seed=42)
- Save validation checkpoints periodically and monitor final answer accuracy and COT quality.

---

## 4. Inference & Demo ▶️
Prompt template to elicit chain-of-thought:
```
System: You are an assistant skilled in audio reasoning. Please list detailed reasoning steps (Chain-of-Thought) before giving the final answer.
User: Question: <question text>
Audio description: <transcript or feature summary>
Please reason step-by-step and provide the final answer.
```

Recommended inference hyperparameters:
- temperature: 0.0 - 0.7 (0.0 for deterministic answers)
- top_p: 0.9
- max_new_tokens: 256-512
- stop_sequences: ["\n\n", "Answer:"]

Example inference command:
```bash
python infer.py --model ./checkpoints/finetuned_cot --prompt_file examples/prompt.jsonl --temperature 0.2 --out predictions.jsonl
```

Self-consistency: sample multiple COT outputs and majority-vote final answers for robustness.

---

## 5. Evaluation ✅
Automatic metrics suggestions:
- Final answer accuracy (primary metric).
- COT quality: BLEU/ROUGE/BERTScore vs reference CoT, plus manual annotation.
- Self-consistency gain: compare majority-vote accuracy from multiple samples.

Example evaluation script:
```bash
python eval.py --pred predictions.jsonl --gold data/test.jsonl --metrics accuracy,bleu,rouge
```

Manual evaluation:
- Randomly sample 200 examples for human review of COT correctness.

---

## 6. Model Card & Usage
Key information:
- Model: Qwen3-ommi-thinking-30B fine-tuned with COT.
- License: include base model and training data license details.

Limitations & risks:
- Sensitive to noisy audio transcription/features.
- COT may contain unreliable or hallucinated steps; human review recommended for critical cases.

Disclaimer:
- Do not use this model for clinical, legal, or other high-risk decision-making without thorough validation and compliance.

---

## 7. Reproducibility 🔁
Best practices:
1. Fix random seed and record commit hash.
2. List environment dependencies (provide `environment.yml` or `requirements.txt`).

Example environment dependencies:
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

## 8. FAQ ❓
Q: How to train 30B with memory constraints?
A: Use LoRA/PEFT, 8-bit optimization (bitsandbytes), Deepspeed ZeRO, or distributed multi-GPU.

Q: How to automatically score COT?
A: Use BLEU/ROUGE/BERTScore approximations, but human annotation or task-specific rules are often necessary.

