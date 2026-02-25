# SFT‑Qwen3‑Ommi for Audio-Reasoning Challenge

## Overview

This repository captures the complete workflow for fine-tuning the **Qwen3-ommi-thinking-30B** model with **Chain‑of‑Thought (CoT)** supervision on the **Audio‑Reasoner** dataset using the **Llamaractory** framework. It covers data preparation, model training, inference, evaluation, and guidelines for reproducibility.

---

## Table of Contents

1. [Project Summary](#project-summary)
2. [Data & Preprocessing](#data--preprocessing)
3. [Fine-tuning Configuration & Example Commands](#fine-tuning-configuration--example-commands)
4. [Inference & Demo](#inference--demo)
5. [Evaluation](#evaluation)
6. [Model Card & Usage Notes](#model-card--usage-notes)
7. [Reproducibility](#reproducibility)
8. [FAQs](#faqs)
9. [Acknowledgements](#acknowledgements)
10. [Citations](#citations)

---

## 1. Project Summary

A step-by-step guide for adapting Qwen3‑ommi-thinking‑30B to the Audio‑Reasoner challenge. The repository is intended as a reference implementation for researchers and practitioners seeking to apply CoT fine‑tuning to multimodal audio reasoning tasks.

## 2. Data & Preprocessing
- **Dataset:** [Audio-Reasoner-CoTA](https://huggingface.co/datasets/zhifeixie/Audio-Reasoner-CoTA).
- **Storage format:** Apache Parquet (columnar) for efficient I/O.
- **Preprocessing pipeline:**
  1. Execute `python demo.py` to extract and store audio samples with their corresponding CoT labels.
  2. Run `python qwen3_audio_think_sft.py` to convert the labels into the format expected by Llamafactory.

The processed data should be organized under a `DATA_DIR` directory, preserving the original structure.

Example training sample (JSON-like):
```
{"instruction": "Which of the following scenarios best fits the context of the audio? (a) A heartfelt apology after a misunderstanding. (b) A doctor reassuring a patient about their health. (c) An exasperated response during a disagreement. (d) A friendly suggestion to move on from a minor inconvenience.", "input": "<audio>", "output": "<think>\n<PLANNING>\nThe question asks to identify the most fitting scenario for the audio clip out of four options. I need to carefully listen to the audio, analyze the speaker's tone and the content of their utterance, and then compare that analysis to each scenario to determine which one best aligns with the audio's characteristics. I'll particularly focus on the emotional cues and the context implied by the spoken words.</PLANNING>\n<CAPTION>\nThe audio presents a single utterance: \"Look, forget about it, okay?\". The speaker's delivery exhibits clear signs of frustration, anger, and dismissiveness. The phrase is delivered sharply, indicating an unwillingness to continue a conversation or address a specific issue. The addition of \"okay?\" sounds less like a genuine question and more like a demand for agreement or a forceful way to conclude the matter.</CAPTION>\n<REASONING>\nLet's analyze each option:\n\n*   **(a) A heartfelt apology after a misunderstanding:** The tone in the audio is abrasive and dismissive, which is the opposite of a heartfelt apology. An apology would involve remorse and a desire to reconcile, neither of which are present here.\n\n*   **(b) A doctor reassuring a patient about their health:** While a doctor's reassurance can sometimes be firm, it usually contains elements of empathy and concern. The audio lacks any such comforting or reassuring qualities. The speaker is impatient and unwilling to engage, which does not fit a doctor-patient scenario.\n\n*   **(c) An exasperated response during a disagreement:** The tone and content of the utterance perfectly align with this scenario. The phrase \"Look, forget about it, okay?\" is a strong indication that the speaker is tired of arguing or discussing something and wants to end the conversation abruptly. The exasperation is evident in the speaker's voice.\n\n*   **(d) A friendly suggestion to move on from a minor inconvenience:** Although the speaker does suggest forgetting something, the delivery is not friendly. The tone is forceful, and the speaker seems frustrated rather than simply offering a gentle suggestion. The \"okay?\" tag adds to the dismissive effect.\n\nTherefore, based on the analysis, option (c) is the most fitting scenario.</REASONING>\n<SUMMARY>\nThe audio clip features a dismissive and angry tone, strongly suggesting exasperation. Considering all options, the context that best fits the audio is (c) An exasperated response during a disagreement.</SUMMARY>\n</think>\n<RESPONSE>\nThe audio conveys a strong sense of anger and dismissiveness with the utterance, \"Look, forget about it, okay?\". The tone is not apologetic, reassuring, or friendly. Therefore, the scenario that best fits the context of the audio is an exasperated response during a disagreement. The answer is C.</RESPONSE>\n", "system": "You are Qwen3-Audio-Think, a multimodal reasoning model. You should carefully analyze audio inputs step by step and provide accurate final answers.", "history": [], "audio_path": "/data/wanghh/challenge/meld/wavs/0000001.wav"}
```

---

## 3. Fine-tuning Configuration & Example Commands 🔧

### Dependencies (suggested)
- Python==3.11
- torch==2.10
- transformers==4.57.1
- accelerate / deepspeed
- bitsandbytes 
- llama/llamaractory
- vllm
- flash-attn
*Using Docker is recommended; the author successfully ran a container based on `11.8.0-cudnn8-runtime-ubuntu22.04`. Pick a CUDA version compatible with your Ubuntu release. An image will be provided shortly.*

Example training configuration (YAML template):
```yaml
### model
model_name_or_path: /home/wanghh/Qwen3-Omni-30B-A3B-Thinking
quantization_bit: 4  # choices: [8 (bnb/hqq/eetq), 4 (bnb/hqq), 3 (hqq), 2 (hqq)]
quantization_method: bnb  # choices: [bnb, hqq, eetq]
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 4
lora_target: all

### dataset
dataset: qwen3-audio-thinking
template: qwen3
cutoff_len: 1024
max_samples: 379288
preprocessing_num_workers: 16
dataloader_num_workers: 2
group_by_length: true
### output
output_dir: saves/qwen3-30b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
gradient_checkpointing: true

```

Example training commands:
```bash
# single-node multi-GPU (example)
llamafactory-cli train examples/train_qlora/qwen3_lora_sft_otfq.yaml
```

**Training notes:**

- The experiment used four 24 GB RTX 3090 GPUs, necessitating 4‑bit quantization. Enable gradient accumulation and tune batch size and learning rate accordingly.
- Set a fixed random seed (e.g., 42) to ensure reproducible results.

---

## 4. Inference command ▶️
First, merge the models with:
```
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```
Then run the baseline inference script to evaluate the capabilities of the SFT Qwen3-Ommi-Thinking-30B on the MMAR benchmark, focusing on Chain-of-Thought (CoT) reasoning in complex acoustic scenarios:
```
python infer_single_model_baseline.py \
   --qwen3_omni_model_name_or_path PATH/TO/Qwen3-Omni-30B-A3B-Thinking \
   --dataset_meta_path PATH/TO/MMAR-meta.json \
   --dataset_audio_prefix PATH/TO/MMAR/audio \
   --flash_attention True \
   --output_dir outputs/single_model_baseline \
   --max_new_tokens 1024
```
```
*Note: the inference script operates on the SFT model produced by the fine-tuning process described earlier.*
```


---

## 5. Evaluation

The primary evaluation is conducted on the MMAR benchmark. Key metrics include:

- **Final-answer accuracy** (overall and modality-wise).
- **Chain-of-Thought quality**, assessed via automated metrics (BLEU/ROUGE/BERTScore) and occasional manual review.
- **Self-consistency** gain when sampling multiple reasoning chains.

Below are the absolute accuracies for the baseline and the 4‑bit quantized SFT model:

| Models                              | Avg    | Sound  | Music  | Speech | Sound-Music | Sound-Speech | Music-Speech | Sound-Music-Speech |
|-------------------------------------|--------|--------|--------|--------|-------------|--------------|--------------|--------------------|
| Qwen3-ommi-thinking-30B             | 66.60% | 61.82% | 41.25% | 78.57% | 63.64%      | 76.61%       | 70.73%       | 66.67%             |
| SFT-Qlora-Qwen3-ommi-thinking-30B   | 57.30% | 50.91% | 41.26% | 64.97% | 58.06%      | 67.89%       | 56.10%       | 50.00%             |

| Models                              | Avg    | Sound  | Music  | Speech | Sound-Music | Sound-Speech | Music-Speech | Sound-Music-Speech |
|-------------------------------------|--------|--------|--------|--------|-------------|--------------|--------------|--------------------|
| Qwen3-ommi-thinking-30B             | 66.60% | 61.82% | 41.25% | 78.57% | 63.64%      | 76.61%       | 70.73%       | 66.67%             |
| SFT-Qlora-Qwen3-ommi-thinking-30B   | 57.30% | 50.91% | 41.26% | 64.97% | 58.06%      | 67.89%       | 56.10%       | 50.00%             |



## 6. Model Card & Usage Notes

**Model:** fine-tuned Qwen3-ommi-thinking-30B with CoT supervision.

**License:** please consult the licenses of the base model and the Audio-Reasoner dataset. Ensure compliance with both when distributing or deploying the tuned model.

**Limitations & Risks:**
- Performance degrades with noisy or inaccurate audio transcriptions.
- Generated reasoning chains may contain hallucinations; critical applications should include human oversight.

**Disclaimer:** not intended for clinical, legal, or other high-stakes decision-making without extensive validation.

---

## 7. Reproducibility

Follow these practices to reproduce experiments:
1. Fix random seeds and record code/model commit hashes.
2. Document environment dependencies (e.g., via an `environment.yml` or `requirements.txt`).
3. Archive training logs, configs, and tokenizer files.

---

## 8. FAQs

**Q:** How can I train the 30B model with limited GPU memory?
**A:** Use parameter-efficient methods such as LoRA or PEFT, 8-bit quantization (bitsandbytes), Deepspeed ZeRO, or distribute across multiple GPUs.

**Q:** Is there an automated way to score Chain-of-Thought responses?
**A:** You can apply BLEU/ROUGE/BERTScore as proxies, but human annotation or task-specific heuristics often yield more reliable results.

---

## 9. Acknowledgements 🔁
This work builds on the **Llamafactory** framework and the [Audio-Reasoning-Challenge-Baselines](https://github.com/Audio-Reasoning-Challenge/Audio-Reasoning-Challenge-Baselines) repository. If these resources help you, please consider giving them a star.

---

## 10. Citations

Please cite the following sources when referencing this project:

- Audio Reasoning Challenge: https://audio-reasoning-challenge.github.io/
- Llamafactory: https://github.com/hiyouga/LlamaFactory
- Qwen3 model: (insert appropriate paper/URL)

---

---
