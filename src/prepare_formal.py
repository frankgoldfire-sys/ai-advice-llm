import random
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

QUESTIONS_PER_SUBJECT = 40
RANDOM_SEED = 20261011
LETTERS = ["A", "B", "C", "D"]


project_root = Path(__file__).resolve().parents[1]

pilot_path = (
    project_root
    / "data"
    / "pilot_20.csv"
)

output_path = (
    project_root
    / "data"
    / "formal_200.csv"
)


def normalize_question(text):
    """
    去掉首尾空格，并把连续空格和换行统一起来。
    用于判断两道题是否相同。
    """
    return " ".join(str(text).split())


# 读取Pilot题目，用于排除已经使用过的20道题
pilot_data = pd.read_csv(pilot_path)

pilot_questions = {
    normalize_question(question)
    for question in pilot_data["question"]
}

rows = []

print("开始准备正式实验题目……")
print(f"固定随机种子：{RANDOM_SEED}")


for subject_number, subject in enumerate(SUBJECTS):
    print()
    print(f"正在读取：{subject}")

    dataset = load_dataset(
        "cais/mmlu",
        subject,
        split="test",
    )

    # 找到所有没有在Pilot中使用过的原始行号
    available_indices = [
        source_row
        for source_row, item in enumerate(dataset)
        if normalize_question(item["question"])
        not in pilot_questions
    ]

    print(
        f"排除Pilot后，可用题目："
        f"{len(available_indices)}"
    )

    if len(available_indices) < QUESTIONS_PER_SUBJECT:
        raise RuntimeError(
            f"{subject}剩余题目不足"
            f"{QUESTIONS_PER_SUBJECT}道。"
        )

    # 每个学科使用不同但固定的随机种子
    subject_seed = RANDOM_SEED + subject_number

    random_generator = random.Random(subject_seed)

    selected_indices = random_generator.sample(
        available_indices,
        QUESTIONS_PER_SUBJECT,
    )

    # 排序只影响保存顺序，不改变随机抽样结果
    selected_indices.sort()

    for source_row in selected_indices:
        item = dataset[source_row]

        if len(item["choices"]) != 4:
            raise ValueError(
                f"{subject}第{source_row}行"
                "不是四选一题目。"
            )

        answer_number = int(item["answer"])

        if answer_number not in [0, 1, 2, 3]:
            raise ValueError(
                f"{subject}第{source_row}行"
                "的答案编号无效。"
            )

        rows.append(
            {
                "question_id": (
                    f"{subject}_test_"
                    f"{source_row:04d}"
                ),
                "source_dataset": "cais/mmlu",
                "subject": subject,
                "source_split": "test",
                "source_row": source_row,
                "sampling_seed": subject_seed,
                "question": item["question"],
                "choice_A": item["choices"][0],
                "choice_B": item["choices"][1],
                "choice_C": item["choices"][2],
                "choice_D": item["choices"][3],
                "correct_answer": (
                    LETTERS[answer_number]
                ),
            }
        )

    print(
        f"已抽取：{len(selected_indices)}道"
    )


formal_data = pd.DataFrame(rows)


# 检查1：总题数必须是200
expected_total = (
    len(SUBJECTS)
    * QUESTIONS_PER_SUBJECT
)

if len(formal_data) != expected_total:
    raise RuntimeError(
        f"题目数量错误："
        f"实际{len(formal_data)}道，"
        f"应为{expected_total}道。"
    )


# 检查2：正式题目不能出现重复
normalized_formal_questions = formal_data[
    "question"
].map(normalize_question)

if normalized_formal_questions.duplicated().any():
    raise RuntimeError(
        "正式题目中发现重复题目。"
    )


# 检查3：正式题目不能与Pilot重叠
formal_question_set = set(
    normalized_formal_questions
)

overlap = (
    formal_question_set
    & pilot_questions
)

if overlap:
    raise RuntimeError(
        f"正式题目与Pilot重叠："
        f"{len(overlap)}道。"
    )


formal_data.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)


print()
print("=" * 60)
print("正式实验题目准备完成")
print(f"题目总数：{len(formal_data)}")
print(f"与Pilot重叠：{len(overlap)}")
print(f"保存位置：{output_path}")

print()
print("各学科题目数量：")
print(
    formal_data["subject"]
    .value_counts()
    .reindex(SUBJECTS)
)

print()
print("前5道题：")
print(
    formal_data[
        [
            "question_id",
            "subject",
            "source_row",
            "correct_answer",
        ]
    ].head()
)