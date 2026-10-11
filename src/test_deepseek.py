import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"

load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有读取到DEEPSEEK_API_KEY，请检查项目根目录中的.env文件。"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

print("正在调用DeepSeek API……")

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "system",
            "content": "Answer briefly and directly.",
        },
        {
            "role": "user",
            "content": "What is 2 + 2? Reply with only the answer.",
        },
    ],
    max_tokens=20,
    stream=False,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)

answer = response.choices[0].message.content

print()
print("DeepSeek回答：")
print(answer)

print()
print("Token使用情况：")
print(response.usage)