import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "deepseek-flash"
RANDOM_SEED = 42
LETTERS = ["A", "B", "C", "D"]

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "pilot_20.csv"
env_path = project_root / ".env"
output_path = (
    project_root
    / "results"
    / "pilot_20_six_conditions.csv"
)

load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有读取到DEEPSEEK_API_KEY。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

system_message = {
    "role": "system",
    "content": (
        "You are a decision-making agent. "
        "Return valid JSON only in the format "
        '{"answer": "A"}. '
        "The answer must be exactly one of A, B, C, or D."
    ),
}


def call_model(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={
            "type": "json_object"
        },
        temperature=0,
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
    answer = parsed_response["answer"].strip().upper()

    if answer not in LETTERS:
        raise ValueError(f"模型返回了无效答案：{answer}")

    metadata = {
        "raw_response": raw_response,
        "returned_model": response.model,
        "system_fingerprint": response.system_fingerprint,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, metadata


def create_result(
    question,
    condition,
    final_answer,
    metadata,
    advice_answer=None,
    initial_answer=None,
):
    correct_answer = (
        question["correct_answer"].strip().upper()
    )

    if initial_answer is None:
        answer_changed = None
        corrected = None
        corrupted = None
    else:
        answer_changed = final_answer != initial_answer

        if initial_answer != correct_answer:
            corrected = final_answer == correct_answer
        else:
            corrected = None

        if initial_answer == correct_answer:
            corrupted = final_answer != correct_answer
        else:
            corrupted = None

    return {
        "question_id": question["question_id"],
        "subject": question["subject"],
        "condition": condition,
        "correct_answer": correct_answer,
        "initial_answer": initial_answer,
        "advice_answer": advice_answer,
        "final_answer": final_answer,
        "final_correct": final_answer == correct_answer,
        "matches_advice": (
            final_answer == advice_answer
            if advice_answer is not None
            else None
        ),
        "answer_changed": answer_changed,
        "corrected": corrected,
        "corrupted": corrupted,
        **metadata,
    }


data = pd.read_csv(data_path)
results = []

random_generator = random.Random(RANDOM_SEED)

total_questions = len(data)

for question_number, (_, question) in enumerate(
    data.iterrows(),
    start=1,
):
    print()
    print("=" * 60)
    print(
        f"正在运行题目 "
        f"{question_number}/{total_questions}："
        f"{question['question_id']}"
    )

    correct_answer = (
        question["correct_answer"].strip().upper()
    )

    question_text = f"""
Question:
{question["question"]}

Options:
A. {question["choice_A"]}
B. {question["choice_B"]}
C. {question["choice_C"]}
D. {question["choice_D"]}
""".strip()

    # 1. Baseline
    print("  1/6 Baseline")

    baseline_messages = [
        system_message,
        {
            "role": "user",
            "content": (
                question_text
                + "\n\nAnswer independently "
                + "without external advice."
            ),
        },
    ]

    baseline_answer, baseline_metadata = call_model(
        baseline_messages
    )

    # 2. Reanswer Control
    print("  2/6 Reanswer Control")

    reanswer_messages = baseline_messages + [
        {
            "role": "assistant",
            "content": baseline_metadata["raw_response"],
        },
        {
            "role": "user",
            "content": (
                "No new information is provided. "
                "Please reconsider the question independently. "
                "You may keep or revise your initial answer. "
                "Return your final decision as valid JSON."
            ),
        },
    ]

    reanswer_answer, reanswer_metadata = call_model(
        reanswer_messages
    )

    # 为本题生成一个错误建议。
    # 错误建议既不能是标准答案，
    # 也不能与模型的初始答案相同。
    wrong_candidates = [
        letter
        for letter in LETTERS
        if letter != correct_answer
        and letter != baseline_answer
    ]

    wrong_advice = random_generator.choice(
        wrong_candidates
    )

    # 3. Correct-first
    print("  3/6 Correct-first")

    correct_first_messages = [
        system_message,
        {
            "role": "user",
            "content": (
                question_text
                + "\n\nExternal advice:\n"
                + "Another AI system recommends option "
                + f"{correct_answer}.\n\n"
                + "Make your final decision."
            ),
        },
    ]

    (
        correct_first_answer,
        correct_first_metadata,
    ) = call_model(correct_first_messages)

    # 4. Wrong-first
    print("  4/6 Wrong-first")

    wrong_first_messages = [
        system_message,
        {
            "role": "user",
            "content": (
                question_text
                + "\n\nExternal advice:\n"
                + "Another AI system recommends option "
                + f"{wrong_advice}.\n\n"
                + "Make your final decision."
            ),
        },
    ]

    (
        wrong_first_answer,
        wrong_first_metadata,
    ) = call_model(wrong_first_messages)

    # 5. Independent → Correct
    print("  5/6 Independent → Correct")

    independent_correct_messages = (
        baseline_messages
        + [
            {
                "role": "assistant",
                "content": baseline_metadata["raw_response"],
            },
            {
                "role": "user",
                "content": (
                    "You have now received external advice. "
                    "Another AI system recommends option "
                    f"{correct_answer}. "
                    "You may keep or revise your initial answer. "
                    "Return your final decision as valid JSON."
                ),
            },
        ]
    )

    (
        independent_correct_answer,
        independent_correct_metadata,
    ) = call_model(independent_correct_messages)

    # 6. Independent → Wrong
    print("  6/6 Independent → Wrong")

    independent_wrong_messages = (
        baseline_messages
        + [
            {
                "role": "assistant",
                "content": baseline_metadata["raw_response"],
            },
            {
                "role": "user",
                "content": (
                    "You have now received external advice. "
                    "Another AI system recommends option "
                    f"{wrong_advice}. "
                    "You may keep or revise your initial answer. "
                    "Return your final decision as valid JSON."
                ),
            },
        ]
    )

    (
        independent_wrong_answer,
        independent_wrong_metadata,
    ) = call_model(independent_wrong_messages)

    question_results = [
        create_result(
            question=question,
            condition="baseline",
            final_answer=baseline_answer,
            metadata=baseline_metadata,
        ),
        create_result(
            question=question,
            condition="reanswer_control",
            final_answer=reanswer_answer,
            metadata=reanswer_metadata,
            initial_answer=baseline_answer,
        ),
        create_result(
            question=question,
            condition="correct_first",
            final_answer=correct_first_answer,
            metadata=correct_first_metadata,
            advice_answer=correct_answer,
        ),
        create_result(
            question=question,
            condition="wrong_first",
            final_answer=wrong_first_answer,
            metadata=wrong_first_metadata,
            advice_answer=wrong_advice,
        ),
        create_result(
            question=question,
            condition="independent_correct",
            final_answer=independent_correct_answer,
            metadata=independent_correct_metadata,
            advice_answer=correct_answer,
            initial_answer=baseline_answer,
        ),
        create_result(
            question=question,
            condition="independent_wrong",
            final_answer=independent_wrong_answer,
            metadata=independent_wrong_metadata,
            advice_answer=wrong_advice,
            initial_answer=baseline_answer,
        ),
    ]

    results.extend(question_results)

    # 每完成一道题就保存一次。
    # 即使后面发生错误，已经完成的数据也不会丢失。
    pd.DataFrame(results).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"  本题完成：初始答案={baseline_answer}，"
        f"标准答案={correct_answer}，"
        f"错误建议={wrong_advice}"
    )


results_df = pd.DataFrame(results)

print()
print("=" * 60)
print("20题 Pilot 实验完成")
print(f"共生成 {len(results_df)} 条实验记录")
print(f"结果保存至：{output_path}")
print()

condition_order = [
    "baseline",
    "reanswer_control",
    "correct_first",
    "wrong_first",
    "independent_correct",
    "independent_wrong",
]

print("各组最终正确率：")

for condition in condition_order:
    condition_data = results_df[
        results_df["condition"] == condition
    ]

    accuracy = condition_data["final_correct"].mean()

    print(
        f"{condition}: "
        f"{accuracy:.1%} "
        f"({condition_data['final_correct'].sum()}"
        f"/{len(condition_data)})"
    )

print()
print(
    "总Token使用量："
    f"{results_df['total_tokens'].sum()}"
)