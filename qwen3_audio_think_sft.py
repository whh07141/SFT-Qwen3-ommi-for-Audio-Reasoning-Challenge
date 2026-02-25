import os
import json
from tqdm import tqdm

AUDIO_DIR = "wavs"
JSON_DIR  = "labels"
OUT_JSONL = "qwen3_audio_think_sft.jsonl"

SYSTEM_PROMPT = (
    "You are Qwen3-Audio-Think, a multimodal reasoning model. "
    "You should carefully analyze audio inputs step by step "
    "and provide accurate final answers."
)

def normalize_think(text: str) -> str:
    text = text.replace("<THINK>", "<think>")
    text = text.replace("</THINK>", "</think>")
    return text

with open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for fname in tqdm(os.listdir(JSON_DIR)):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(JSON_DIR, fname)
        wav_path = os.path.join('/data/wanghh/challenge/meld/',AUDIO_DIR, fname.replace(".json", ".wav"))

        if not os.path.exists(wav_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        user = data["user"]          # "<audio>Transcribe sound into Chinese"
        assistant = data["assistant"]

        # 拆 instruction / input
        if user.startswith("<audio>"):
            instruction = user.replace("<audio>", "").strip()
            input_text = "<audio>"
        else:
            instruction = user
            input_text = ""

        assistant = normalize_think(assistant)

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": assistant,
            "system": SYSTEM_PROMPT,
            "history": [],
            "audio_path": wav_path
        }

        fout.write(json.dumps(sample, ensure_ascii=False) +','+ "\n")

print("Done. Saved to", OUT_JSONL)

