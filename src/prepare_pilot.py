from pathlib import Path

import pandas as pd
from datasets import load_dataset


SUBJECTS = [
    "business_ethics",
    "high_school_microeconomics",
    "high_school_macroeconomics",
    "professional_law",
    "college_computer_science",
]

QUESTIONS_PER_SUBJECT = 4
RANDOM_SEED = 42
LETTERS = ["A", "B", "C", "D"]

rows = []

print("开始准备20道Pilot题目……")

for subject in SUBJECTS:
    print(f"正在读取：{subject}")

    dataset = load_dataset(
        "cais/mmlu",
        subject,
        split="test",
    )

    sample = dataset.shuffle(seed=RANDOM_SEED).select(
        range(QUESTIONS_PER_SUBJECT)
    )

    for number, item in enumerate(sample, start=1):
        rows.append(
            {
                "question_id": f"{subject}_{number:02d}",
                "subject": subject,
                "question": item["question"],
                "choice_A": item["choices"][0],
                "choice_B": item["choices"][1],
                "choice_C": item["choices"][2],
                "choice_D": item["choices"][3],
                "correct_answer": LETTERS[item["answer"]],
            }
        )

pilot_data = pd.DataFrame(rows)

project_root = Path(__file__).resolve().parents[1]
output_path = project_root / "data" / "pilot_20.csv"

pilot_data.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print()
print(f"成功生成 {len(pilot_data)} 道题目")
print(f"保存位置：{output_path}")

print()
print("前5行数据：")
print(
    pilot_data[
        ["question_id", "subject", "correct_answer"]
    ].head()
)