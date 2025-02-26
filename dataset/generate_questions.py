import openai
import pandas as pd
from tqdm import tqdm
from typing import List
import os
import time  # 新增，用于错误重试时等待时间
import hashlib  # 新增，用于计算问题哈希

# OpenAI配置
openai.api_key = os.getenv("OPENAI_API_KEY")
num_generations = 1000

def generate_question_with_openai(prompt: str) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # 使用有效模型名称
                prompt=prompt,
                temperature=0.7,
                max_tokens=100
            )
            return response['choices'][0]['text'].strip()  # 获取生成的问题文本
        except Exception as e:
            print(f"生成问题时出错, 尝试 {attempt+1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)  # 指数退避
    return None

def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()[:16]

def generate_questions(prompt: str) -> List[dict]:
    unique_set = set()
    questions = []
    pbar = tqdm(total=num_generations, desc="生成问题中")
    while len(unique_set) < num_generations:
        question = generate_question_with_openai(prompt)
        if question:
            hash_val = question_hash(question)
            if hash_val not in unique_set:
                unique_set.add(hash_val)
                questions.append({"question": question, "prompt": prompt})
                pbar.update(1)
    pbar.close()
    return questions

def publish_to_huggingface(dataset: List[dict], dataset_name: str):
    from huggingface_hub import HfApi, HfFolder
    from datasets import Dataset
    
    # 转换为pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # 转换为Hugging Face数据集格式
    hf_dataset = Dataset.from_pandas(df)
    
    # 保存到本地
    hf_dataset.save_to_disk("./generated_dataset")
    
    # 发布到Hugging Face
    api = HfApi()
    token = HfFolder.get_token()
    
    if not token:
        raise ValueError("请设置Hugging Face token!")
        
    try:
        api.create_repo(repo_id=dataset_name, token=token, exist_ok=True)
        api.upload_folder(
            folder_path="./generated_dataset",
            repo_id=dataset_name,
            repo_type="dataset",
            token=token
        )
        print(f"数据集已成功发布到: https://huggingface.co/datasets/{dataset_name}")
    except Exception as e:
        print(f"发布数据集时出错: {e}")

if __name__ == "__main__":
    prompt = """
    请生成一系列关于陪伴和伴侣的深度问题，探索人与人之间关系的多样性和复杂性。问题可以包括：
    1. 关于陪伴在人际关系中的重要性。
    2. 伴侣之间如何平衡个人空间与共同时间。
    3. 在长期关系中，如何保持相互之间的沟通与理解。
    4. 伴侣关系中的信任与支持如何塑造彼此的成长。
    5. 陪伴对情感和心理健康的影响。
    6. 在现代社会中，伴侣关系面临的主要挑战和变化。
    7. 伴侣如何共同应对生活中的重大变化或危机。

    请确保问题涵盖不同的情感层面、社会观念以及心理层面的角度。
    """

    dataset_name = "your-username/open-r1-questions-dataset"  # 修改为你的用户名和数据集名称
    
    questions = generate_questions(prompt)
    if questions:
        publish_to_huggingface(questions, dataset_name)
