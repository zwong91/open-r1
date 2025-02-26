from typing import Dict
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_cot_example(
    example: Dict,
    tokenizer,
):
    thinking_trajectory = example["thinking_trajectories"]
    question = example["question"]
    answer = example["attempt"] 
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {
            "role": "assistant", 
            "content": "<|im_start|>think\n" + "\n".join(thinking_trajectory).strip() + "\n<|im_start|>answer\n" + answer.strip()
        }
    ], tokenize=False)
    return dict(text=text)

def mathcot_sft(upload_data_path: str, num_proc: int,
                download_data_path):

    dataset = load_dataset(download_data_path, download_mode='force_redownload')
    if 'train' in dataset:
        dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    process_example_map = partial(process_cot_example, tokenizer=tokenizer)
    dataset = dataset.map(
        process_example_map,
        num_proc=num_proc,
        desc="Tokenizing SFT data",
    )
    dataset.push_to_hub(upload_data_path)

if __name__ == "__main__":
    mathcot_sft(download_data_path="simplescaling/s1K",
                upload_data_path="simplescaling/s1K_tokenized", 
                num_proc=20)
