import os
import json
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
ds1 = load_dataset("/data/wanghh/challenge/Audio-Reasoner", "complex_audio")
out_root ="complex_audio"
wav_dir = os.path.join(out_root, "wavs")
label_dir = os.path.join(out_root, "labels")
os.makedirs(wav_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

global_idx = 0

for part_name, dataset in (ds1.items()):
    print(f"processing {part_name}")

    for sample in tqdm(dataset):
        audio = sample["audio"]
    
        
        if isinstance(audio, dict):
            wave = audio["array"]
            sr = audio["sampling_rate"]
        else:
            wave = np.array(audio)
            sr = 16000

        wav_path = os.path.join(wav_dir, f"{global_idx:07d}.wav")
        sf.write(wav_path, wave, sr)

        label = {
            "id": global_idx,
            "part": part_name,
            "user": sample["user"],
            "assistant": sample["assistant"]
        }

        label_path = os.path.join(label_dir, f"{global_idx:07d}.json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        global_idx += 1

