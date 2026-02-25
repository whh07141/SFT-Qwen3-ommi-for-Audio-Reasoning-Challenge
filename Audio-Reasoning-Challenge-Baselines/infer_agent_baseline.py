from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json
import logging

import torch
from transformers import (
    HfArgumentParser,
    BatchFeature,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerConfig,
)
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

root_logger = logging.getLogger()

USE_AUDIO_IN_VIDEO = True


@dataclass
class InferConfig:
    qwen3_omni_model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to a local checkpoint directory OR a model ID on the Hugging-Face Hub, "
                "e.g.  './checkpoints/qwen3_omni_moe'."
            )
        }
    )
    qwen3_model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to a local checkpoint directory OR a model ID on the Hugging-Face Hub, "
                "e.g.  './checkpoints/qwen3_moe'."
            )
        }
    )
    dataset_meta_path: str = field(
        metadata={
            "help": (
                "Path to a JSON/JSONL metadata file. Each record must contain "
                "'id', 'audio_path', 'question', and 'choices' fields."
            )
        }
    )
    dataset_audio_prefix: str = field(
        default="",  
        metadata={
            "help": (
                "Prefix prepended to every 'audio_path' found in the metadata. "
            )
        },
    )
    dataset_start: int = field(
        default=0,
        metadata={
            "help": "The start of the dataset to infer. Together with dataset_end, defines the interval [dataset_start, dataset_end)."
        },
    )
    dataset_end: Optional[int] = field(
        default=None,
        metadata={
            "help": "The end of the dataset to infer. Together with dataset_start, defines the interval [dataset_start, dataset_end)."
        },
    )
    flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use flash attention."}
    )
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "The output directory where the model predictions will be written."
        },
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum number of new tokens to generate. If None, the model will generate as many tokens as it can."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "The repetition penalty to apply to the generated tokens. If None, no repetition penalty will be applied."
        },
    )
    save_steps: int = field(
        default=8,
        metadata={"help": "The number of steps to save the model predictions."},
    )

def load_qwen3_omni(
    model_name_or_path: str,
    flash_attention: bool = False,
):
    model_config = Qwen3OmniMoeThinkerConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_name_or_path,
    )

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=model_config,
        torch_dtype=model_config.torch_dtype,
        trust_remote_code=True,
        weights_only=False,
        attn_implementation="flash_attention_2" if flash_attention else None,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    print(
        f"Loaded Qwen3-Omni from {model_name_or_path}, dtype {model_config.torch_dtype}"
    )
    return model, processor

def load_qwen3(
    model_name_or_path: str,
    flash_attention: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if flash_attention else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    return model, tokenizer

def prepare_mmar_inputs_for_qwen3_omni(
    processor: Qwen3OmniMoeProcessor,
    sample: Dict,
):
    audio_path = sample["audio_path"]
    question = (
        sample["question"]
        + "\nSelect one option from the provided choices:\n"
        + "\n".join(sample["choices"])
    )
    prompt_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a multi-task audio understanding and reasoning model.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question},
            ],
        },
    ]
    audio, *_ = process_mm_info(prompt_messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    prompt_text = processor.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    prompt_example = processor(
        text=prompt_text,
        audio=audio,
        images=None,
        videos=None,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        min_pixels=processor.video_processor.min_pixels,  # 2 * 2 * 28 * 28
        max_pixels=processor.video_processor.max_pixels,  # 128 * 128 * 28 * 28
        sampling_rate=processor.feature_extractor.sampling_rate,  # 16000
    )
    return prompt_example

def prepare_inputs_for_qwen3(
    tokenizer,
    sample: Dict,
    thinking_content: str,
    content: str,
):
    prompt = (
        "Question: " + sample["question"] + 
        "Options: " + "|".join(sample["choices"]) +
        "Original thinking prediction: " + thinking_content +
        "Original answer prediction: " + content +
        "Closely analyze the provided question and options. Refine the thinking process and the final answer. Your output must be a JSON object with the following structure:" +
        """
        {
            "thinking_prediction": "Refined reasoning process.",
            "answer_prediction": "Refined final answer."
        }
        """
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # use non-thinking mode
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    return model_inputs

def generate_qwen3_omni(
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    processor: Qwen3OmniMoeProcessor,
    batch: BatchFeature,
    max_new_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
):
    batch_on_device = batch.to(model.device).to(model.dtype)
    output = model.generate(
        **batch_on_device,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        do_sample=False,
        num_beams=1,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    ).cpu()
    output_ids = output[0][len(batch_on_device.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = processor.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content

def generate_qwen3(
    model: AutoModelForCausalLM,
    tokenizer,
    batch: BatchFeature,
    max_new_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
):
    generated_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        do_sample=False,
        num_beams=1,
    )
    output_ids = generated_ids[0][len(batch.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    content_dict = json.loads(content)
    assert "thinking_prediction" in content_dict
    assert "answer_prediction" in content_dict
    return content_dict

def load_mmar(
    dataset_meta_path: str,
    dataset_audio_prefix: str,
    start: int = 0,
    end: Optional[int] = None,
):
    with open(dataset_meta_path, "r", encoding="utf-8") as fin:
        sample_list = json.load(fin)
        dataset_slice = slice(start, end)
        sample_list = sample_list[dataset_slice]

        for i in range(len(sample_list)):
            real_audio_path = os.path.realpath(
                os.path.join(dataset_audio_prefix, sample_list[i]["audio_path"])
            )
            sample_list[i]["audio_path"] = real_audio_path
    return sample_list

class ResultFile:
    def __init__(self, path: str, save_steps: int = 8):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.records = []
        self.save_steps = save_steps

    def add_record(self, record: dict):
        self.records.append(record)
        if len(self.records) >= self.save_steps:
            self.flush()

    def flush(self):
        if len(self.records) == 0:
            return

        with open(self.path, "a", encoding="utf-8") as jsonl_file:
            for record in self.records:
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()

        self.records.clear()

    def close(self):
        self.flush()

def infer(config: InferConfig):
    output_jsonl_path = os.path.join(config.output_dir, f"prediction.jsonl")
    result_file = ResultFile(output_jsonl_path, save_steps=config.save_steps)

    qwen3_omni_model, qwen3_omni_processor = load_qwen3_omni(
        model_name_or_path=config.qwen3_omni_model_name_or_path,
        flash_attention=config.flash_attention,
    )

    qwen3_model, qwen3_tokenizer = load_qwen3(
        model_name_or_path=config.qwen3_model_name_or_path,
        flash_attention=config.flash_attention,
    )

    sample_list = load_mmar(
        dataset_meta_path=config.dataset_meta_path,
        dataset_audio_prefix=config.dataset_audio_prefix,
        start=config.dataset_start,
        end=config.dataset_end,
    )

    for i, sample in tqdm(
        enumerate(sample_list), total=len(sample_list), desc="Infering"
    ):
        with torch.no_grad():
            batch = prepare_mmar_inputs_for_qwen3_omni(qwen3_omni_processor, sample)

            try:
                thinking_content, content = generate_qwen3_omni(
                    qwen3_omni_model,
                    qwen3_omni_processor,
                    batch,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=config.repetition_penalty,
                )

                batch = prepare_inputs_for_qwen3(qwen3_tokenizer, sample, thinking_content, content)

                content_dict = generate_qwen3(
                    qwen3_model,
                    qwen3_tokenizer,
                    batch,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=config.repetition_penalty,
                )
                result = {
                    "id": sample["id"],
                    **content_dict,
                }
                print(result)
                result_file.add_record(result)
            except Exception as e:
                logger.exception(
                    "An unexpected error occurred during inference: %s", str(e)
                )
                logger.error("Error processing sample (sample: %s)", str(sample))
                torch.cuda.empty_cache()

    result_file.close()

def main():
    parser = HfArgumentParser([InferConfig])
    (infer_config,) = parser.parse_args_into_dataclasses()
    infer(infer_config)


if __name__ == "__main__":
    main()