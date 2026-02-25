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
- Dataset: [Audio-Reasoner-CoTA](https://huggingface.co/datasets/zhifeixie/Audio-Reasoner-CoTA).
- Format: the dataset is stored in Apache Parquet columnar format.
- Preprocessing:
  - Run `python demo.py` to save audio data along with labels containing full chains of thought.
  - Run `python qwen3_audio_think_sft.py` to generate labels in a format consumable by Llamafactory.

Example training sample (JSON-like):
```
{"instruction": "Which of the following scenarios best fits the context of the audio? (a) A heartfelt apology after a misunderstanding. (b) A doctor reassuring a patient about their health. (c) An exasperated response during a disagreement. (d) A friendly suggestion to move on from a minor inconvenience.", "input": "<audio>", "output": "<think>\n<PLANNING>\nThe question asks to identify the most fitting scenario for the audio clip out of four options. I need to carefully listen to the audio, analyze the speaker's tone and the content of their utterance, and then compare that analysis to each scenario to determine which one best aligns with the audio's characteristics. I'll particularly focus on the emotional cues and the context implied by the spoken words.</PLANNING>\n<CAPTION>\nThe audio presents a single utterance: \"Look, forget about it, okay?\". The speaker's delivery exhibits clear signs of frustration, anger, and dismissiveness. The phrase is delivered sharply, indicating an unwillingness to continue a conversation or address a specific issue. The addition of \"okay?\" sounds less like a genuine question and more like a demand for agreement or a forceful way to conclude the matter.</CAPTION>\n<REASONING>\nLet's analyze each option:\n\n*   **(a) A heartfelt apology after a misunderstanding:** The tone in the audio is abrasive and dismissive, which is the opposite of a heartfelt apology. An apology would involve remorse and a desire to reconcile, neither of which are present here.\n\n*   **(b) A doctor reassuring a patient about their health:** While a doctor's reassurance can sometimes be firm, it usually contains elements of empathy and concern. The audio lacks any such comforting or reassuring qualities. The speaker is impatient and unwilling to engage, which does not fit a doctor-patient scenario.\n\n*   **(c) An exasperated response during a disagreement:** The tone and content of the utterance perfectly align with this scenario. The phrase \"Look, forget about it, okay?\" is a strong indication that the speaker is tired of arguing or discussing something and wants to end the conversation abruptly. The exasperation is evident in the speaker's voice.\n\n*   **(d) A friendly suggestion to move on from a minor inconvenience:** Although the speaker does suggest forgetting something, the delivery is not friendly. The tone is forceful, and the speaker seems frustrated rather than simply offering a gentle suggestion. The \"okay?\" tag adds to the dismissive effect.\n\nTherefore, based on the analysis, option (c) is the most fitting scenario.</REASONING>\n<SUMMARY>\nThe audio clip features a dismissive and angry tone, strongly suggesting exasperation. Considering all options, the context that best fits the audio is (c) An exasperated response during a disagreement.</SUMMARY>\n</think>\n<RESPONSE>\nThe audio conveys a strong sense of anger and dismissiveness with the utterance, \"Look, forget about it, okay?\". The tone is not apologetic, reassuring, or friendly. Therefore, the scenario that best fits the context of the audio is an exasperated response during a disagreement. The answer is C.</RESPONSE>\n", "system": "You are Qwen3-Audio-Think, a multimodal reasoning model. You should carefully analyze audio inputs step by step and provide accurate final answers.", "history": [], "audio_path": "/data/wanghh/challenge/meld/wavs/0000001.wav"}
```

---

## 3. Fine-tuning Config & Example Commands 🔧
**Suggested dependencies (example)**:
- Python==3.11
- torch==2.10
- transformers==4.57.1
- accelerate / deepspeed
- bitsandbytes 
- llama/llamaractory
- vllm
- flash-attn
*It is recommended to use Docker. I ran the container successfully with `11.8.0-cudnn8-runtime-ubuntu22.04`; choose a CUDA version supported by your Ubuntu release. I will upload this image later.*

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

Training notes:
- The training setup uses four 24 GB RTX 3090 cards, which restricts us to 4‑bit quantized training; remember to enable gradient accumulation and choose appropriate batch size/learning rate.
- Fix the random seed for reproducibility (e.g., seed=42).

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
Training notes: the inference run uses the SFT model described above.
```


---

## 5. Result ✅
The following are the results on MMAR for the un-finetuned baseline and the 4‑bit quantized SFT model:

| Models                              | Avg    | Sound  | Music  | Speech | Sound-Music | Sound-Speech | Music-Speech | Sound-Music-Speech |
|-------------------------------------|--------|--------|--------|--------|-------------|--------------|--------------|--------------------|
| Qwen3-ommi-thinking-30B             | 66.60% | 61.82% | 41.25% | 78.57% | 63.64%      | 76.61%       | 70.73%       | 66.67%             |
| SFT-Qlora-Qwen3-ommi-thinking-30B   | 57.30% | 50.91% | 41.26% | 64.97% | 58.06%      | 67.89%       | 56.10%       | 50.00%             |



## 7. Acknowledge 🔁
All of my work is based on Llamafactory and the [Audio-Reasoning-Challenge-Baselines](https://github.com/Audio-Reasoning-Challenge/Audio-Reasoning-Challenge-Baselines) repository. If you find this project useful, please consider giving those projects a star.

---
