import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "deepseek-flash"

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "pilot_20.csv"
env_path = project_root / ".env"
output_path = project_root / "results" / "one_baseline.json"

load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有读取到DEEPSEEK_API_KEY。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

data = pd.read_csv(data_path)
question = data.iloc[0]

prompt = f"""
Answer the following multiple-choice question.

Question:
{question["question"]}

Options:
A. {question["choice_A"]}
B. {question["choice_B"]}
C. {question["choice_C"]}
D. {question["choice_D"]}

Return valid JSON only in this format:
{{"answer": "A"}}

The answer must be exactly one of A, B, C, or D.
""".strip()

print("正在运行第一道Baseline题目……")
print()
print(prompt)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a decision-making agent. "
                "Answer the question independently. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    response_format={
        "type": "json_object"
    },
    max_tokens=50,
    stream=False,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)

raw_response = response.choices[0].message.content
parsed_response = json.loads(raw_response)

model_answer = parsed_response["answer"].strip().upper()
correct_answer = question["correct_answer"].strip().upper()

if model_answer not in ["A", "B", "C", "D"]:
    raise ValueError(
        f"模型返回了无效答案：{model_answer}"
    )

result = {
    "question_id": question["question_id"],
    "condition": "baseline",
    "requested_model": MODEL_NAME,
    "returned_model": response.model,
    "system_fingerprint": response.system_fingerprint,
    "correct_answer": correct_answer,
    "model_answer": model_answer,
    "is_correct": model_answer == correct_answer,
    "raw_response": raw_response,
    "prompt_tokens": response.usage.prompt_tokens,
    "completion_tokens": response.usage.completion_tokens,
    "total_tokens": response.usage.total_tokens,
}

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=2,
    )

print()
print("模型原始回答：")
print(raw_response)

print()
print(f"标准答案：{correct_answer}")
print(f"模型答案：{model_answer}")
print(f"是否正确：{result['is_correct']}")

print()
print(f"结果已保存至：{output_path}")