import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. 实验设置
# ============================================================

MODEL_NAME = "deepseek-flash"
PROMPT_VERSION = "formal_v1"
RANDOM_SEED = 42
LETTERS = ["A", "B", "C", "D"]

# True：只运行1道题，共6次API调用
# False：运行全部200道题，共1200次API调用
TEST_MODE = False
TEST_QUESTION_LIMIT = 1

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5

CONDITIONS = [
    "baseline",
    "reanswer_control",
    "correct_first",
    "wrong_first",
    "independent_correct",
    "independent_wrong",
]


# ============================================================
# 2. 文件路径
# ============================================================

project_root = Path(__file__).resolve().parents[1]

data_path = (
    project_root
    / "data"
    / "formal_200.csv"
)

env_path = project_root / ".env"

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

if TEST_MODE:
    output_path = (
        results_dir
        / "formal_smoke_test.csv"
    )
    run_mode = "smoke_test"
else:
    output_path = (
        results_dir
        / "formal_200_results.csv"
    )
    run_mode = "formal"


# ============================================================
# 3. 初始化DeepSeek客户端
# ============================================================

load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有读取到DEEPSEEK_API_KEY。"
    )

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
        "The answer must be exactly one of "
        "A, B, C, or D."
    ),
}


# ============================================================
# 4. API调用函数：失败后自动重试
# ============================================================

def call_model(messages):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = (
                client.chat.completions.create(
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
            )

            raw_response = (
                response
                .choices[0]
                .message
                .content
            )

            parsed_response = json.loads(
                raw_response
            )

            answer = (
                parsed_response["answer"]
                .strip()
                .upper()
            )

            if answer not in LETTERS:
                raise ValueError(
                    f"模型返回了无效答案："
                    f"{answer}"
                )

            metadata = {
                "requested_model": MODEL_NAME,
                "returned_model": response.model,
                "system_fingerprint": (
                    response.system_fingerprint
                ),
                "prompt_version": PROMPT_VERSION,
                "raw_response": raw_response,
                "prompt_tokens": (
                    response.usage.prompt_tokens
                ),
                "completion_tokens": (
                    response.usage.completion_tokens
                ),
                "total_tokens": (
                    response.usage.total_tokens
                ),
                "request_time_utc": (
                    datetime.now(timezone.utc)
                    .isoformat()
                ),
            }

            return answer, metadata

        except Exception as error:
            last_error = error

            print(
                f"    第{attempt}次调用失败："
                f"{error}"
            )

            if attempt < MAX_RETRIES:
                wait_seconds = (
                    RETRY_WAIT_SECONDS
                    * attempt
                )

                print(
                    f"    等待{wait_seconds}秒"
                    "后重试……"
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"连续{MAX_RETRIES}次调用失败。"
    ) from last_error


# ============================================================
# 5. 构造题目文本
# ============================================================

def build_question_text(question):
    return f"""
Question:
{question["question"]}

Options:
A. {question["choice_A"]}
B. {question["choice_B"]}
C. {question["choice_C"]}
D. {question["choice_D"]}
""".strip()


# ============================================================
# 6. 为每道题稳定生成错误建议
# ============================================================

def choose_wrong_advice(
    question_id,
    correct_answer,
    baseline_answer,
):
    wrong_candidates = [
        letter
        for letter in LETTERS
        if letter != correct_answer
        and letter != baseline_answer
    ]

    seed_text = (
        f"{RANDOM_SEED}:"
        f"{question_id}"
    )

    seed_value = int(
        hashlib.sha256(
            seed_text.encode("utf-8")
        ).hexdigest()[:8],
        16,
    )

    random_generator = random.Random(
        seed_value
    )

    wrong_advice = (
        random_generator.choice(
            wrong_candidates
        )
    )

    return wrong_advice, seed_value


# ============================================================
# 7. 整理单条实验记录
# ============================================================

def create_result(
    question,
    condition,
    condition_order,
    final_answer,
    metadata,
    advice_answer=None,
    initial_answer=None,
    advice_seed=None,
):
    correct_answer = (
        str(question["correct_answer"])
        .strip()
        .upper()
    )

    if initial_answer is None:
        answer_changed = None
        corrected = None
        corrupted = None

    else:
        answer_changed = (
            final_answer != initial_answer
        )

        if initial_answer != correct_answer:
            corrected = (
                final_answer
                == correct_answer
            )
        else:
            corrected = None

        if initial_answer == correct_answer:
            corrupted = (
                final_answer
                != correct_answer
            )
        else:
            corrupted = None

    return {
        "question_id": question["question_id"],
        "source_dataset": (
            question["source_dataset"]
        ),
        "subject": question["subject"],
        "source_split": (
            question["source_split"]
        ),
        "source_row": question["source_row"],
        "run_mode": run_mode,
        "condition": condition,
        "condition_order": condition_order,
        "correct_answer": correct_answer,
        "initial_answer": initial_answer,
        "advice_answer": advice_answer,
        "advice_seed": advice_seed,
        "final_answer": final_answer,
        "final_correct": (
            final_answer == correct_answer
        ),
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


# ============================================================
# 8. 安全保存：先写临时文件，再替换正式文件
# ============================================================

def save_results_safely(
    results,
    destination_path,
):
    results_df = pd.DataFrame(results)

    temporary_path = (
        destination_path
        .with_suffix(".tmp.csv")
    )

    results_df.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temporary_path,
        destination_path,
    )


def append_and_save(
    results,
    result,
):
    results.append(result)

    save_results_safely(
        results,
        output_path,
    )


# ============================================================
# 9. 读取现有结果，实现断点续跑
# ============================================================

def load_existing_results(
    destination_path,
    valid_question_ids,
):
    if not destination_path.exists():
        return [], set()

    existing_data = pd.read_csv(
        destination_path
    )

    if existing_data.empty:
        return [], set()

    existing_question_ids = set(
        existing_data["question_id"]
    )

    unknown_ids = (
        existing_question_ids
        - valid_question_ids
    )

    if unknown_ids:
        raise RuntimeError(
            "结果文件中存在不属于当前"
            "题目集的question_id。"
        )

    if (
        "requested_model"
        in existing_data.columns
    ):
        old_models = set(
            existing_data[
                "requested_model"
            ].dropna()
        )

        if old_models != {MODEL_NAME}:
            raise RuntimeError(
                "现有结果使用的模型"
                "与当前设置不一致。"
            )

    if (
        "prompt_version"
        in existing_data.columns
    ):
        old_versions = set(
            existing_data[
                "prompt_version"
            ].dropna()
        )

        if old_versions != {
            PROMPT_VERSION
        }:
            raise RuntimeError(
                "现有结果使用的提示词版本"
                "与当前设置不一致。"
            )

    progress = (
        existing_data
        .groupby("question_id")
        .agg(
            row_count=(
                "condition",
                "size",
            ),
            condition_count=(
                "condition",
                "nunique",
            ),
        )
    )

    completed_question_ids = set(
        progress[
            (progress["row_count"] == 6)
            & (
                progress[
                    "condition_count"
                ]
                == 6
            )
        ].index
    )

    partial_question_ids = (
        existing_question_ids
        - completed_question_ids
    )

    if partial_question_ids:
        print(
            "发现未完整运行的题目："
            f"{len(partial_question_ids)}道"
        )

        print(
            "这些题目的旧记录将被移除，"
            "然后重新运行完整六组。"
        )

        existing_data = existing_data[
            ~existing_data["question_id"]
            .isin(partial_question_ids)
        ].copy()

        save_results_safely(
            existing_data.to_dict(
                orient="records"
            ),
            destination_path,
        )

    existing_results = (
        existing_data
        .to_dict(orient="records")
    )

    return (
        existing_results,
        completed_question_ids,
    )


# ============================================================
# 10. 读取正式题目
# ============================================================

all_questions = pd.read_csv(data_path)

if all_questions["question_id"].duplicated().any():
    raise RuntimeError(
        "formal_200.csv中存在重复的"
        "question_id。"
    )

if TEST_MODE:
    questions = all_questions.head(
        TEST_QUESTION_LIMIT
    ).copy()

    print("当前为试运行模式")
    print(
        f"只运行前"
        f"{len(questions)}道题。"
    )

else:
    questions = all_questions.copy()

    if len(questions) != 200:
        raise RuntimeError(
            "正式模式要求题目数量为200。"
        )

    print("当前为正式运行模式")
    print("将运行全部200道题。")


valid_question_ids = set(
    questions["question_id"]
)

results, completed_question_ids = (
    load_existing_results(
        output_path,
        valid_question_ids,
    )
)

pending_count = (
    len(questions)
    - len(completed_question_ids)
)

print(
    f"已经完成："
    f"{len(completed_question_ids)}道"
)

print(
    f"等待运行：{pending_count}道"
)

print(f"结果文件：{output_path}")


# ============================================================
# 11. 正式运行六个实验条件
# ============================================================

for question_number, (
    _,
    question,
) in enumerate(
    questions.iterrows(),
    start=1,
):
    question_id = question["question_id"]

    if question_id in completed_question_ids:
        print()
        print(
            f"[{question_number}/"
            f"{len(questions)}] "
            f"{question_id} 已完成，跳过。"
        )
        continue

    print()
    print("=" * 60)

    print(
        f"[{question_number}/"
        f"{len(questions)}] "
        f"{question_id}"
    )

    correct_answer = (
        str(question["correct_answer"])
        .strip()
        .upper()
    )

    question_text = build_question_text(
        question
    )

    # --------------------------------------------------------
    # 1/6 Baseline
    # --------------------------------------------------------

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

    (
        baseline_answer,
        baseline_metadata,
    ) = call_model(baseline_messages)

    append_and_save(
        results,
        create_result(
            question=question,
            condition="baseline",
            condition_order=1,
            final_answer=baseline_answer,
            metadata=baseline_metadata,
        ),
    )

    # --------------------------------------------------------
    # 2/6 Reanswer Control
    # --------------------------------------------------------

    print("  2/6 Reanswer Control")

    reanswer_messages = (
        baseline_messages
        + [
            {
                "role": "assistant",
                "content": (
                    baseline_metadata[
                        "raw_response"
                    ]
                ),
            },
            {
                "role": "user",
                "content": (
                    "No new information is "
                    "provided. Please reconsider "
                    "the question independently. "
                    "You may keep or revise your "
                    "initial answer. Return your "
                    "final decision as valid JSON."
                ),
            },
        ]
    )

    (
        reanswer_answer,
        reanswer_metadata,
    ) = call_model(reanswer_messages)

    append_and_save(
        results,
        create_result(
            question=question,
            condition=(
                "reanswer_control"
            ),
            condition_order=2,
            final_answer=reanswer_answer,
            metadata=reanswer_metadata,
            initial_answer=baseline_answer,
        ),
    )

    # --------------------------------------------------------
    # 生成本题的固定错误建议
    # --------------------------------------------------------

    (
        wrong_advice,
        advice_seed,
    ) = choose_wrong_advice(
        question_id=question_id,
        correct_answer=correct_answer,
        baseline_answer=baseline_answer,
    )

    # --------------------------------------------------------
    # 3/6 Correct-first
    # --------------------------------------------------------

    print("  3/6 Correct-first")

    correct_first_messages = [
        system_message,
        {
            "role": "user",
            "content": (
                question_text
                + "\n\nExternal advice:\n"
                + "Another AI system "
                + "recommends option "
                + f"{correct_answer}.\n\n"
                + "Make your final decision."
            ),
        },
    ]

    (
        correct_first_answer,
        correct_first_metadata,
    ) = call_model(
        correct_first_messages
    )

    append_and_save(
        results,
        create_result(
            question=question,
            condition="correct_first",
            condition_order=3,
            final_answer=(
                correct_first_answer
            ),
            metadata=(
                correct_first_metadata
            ),
            advice_answer=correct_answer,
        ),
    )

    # --------------------------------------------------------
    # 4/6 Wrong-first
    # --------------------------------------------------------

    print("  4/6 Wrong-first")

    wrong_first_messages = [
        system_message,
        {
            "role": "user",
            "content": (
                question_text
                + "\n\nExternal advice:\n"
                + "Another AI system "
                + "recommends option "
                + f"{wrong_advice}.\n\n"
                + "Make your final decision."
            ),
        },
    ]

    (
        wrong_first_answer,
        wrong_first_metadata,
    ) = call_model(
        wrong_first_messages
    )

    append_and_save(
        results,
        create_result(
            question=question,
            condition="wrong_first",
            condition_order=4,
            final_answer=(
                wrong_first_answer
            ),
            metadata=(
                wrong_first_metadata
            ),
            advice_answer=wrong_advice,
            advice_seed=advice_seed,
        ),
    )

    # --------------------------------------------------------
    # 5/6 Independent → Correct
    # --------------------------------------------------------

    print(
        "  5/6 Independent → Correct"
    )

    independent_correct_messages = (
        baseline_messages
        + [
            {
                "role": "assistant",
                "content": (
                    baseline_metadata[
                        "raw_response"
                    ]
                ),
            },
            {
                "role": "user",
                "content": (
                    "You have now received "
                    "external advice. Another AI "
                    "system recommends option "
                    f"{correct_answer}. "
                    "You may keep or revise your "
                    "initial answer. Return your "
                    "final decision as valid JSON."
                ),
            },
        ]
    )

    (
        independent_correct_answer,
        independent_correct_metadata,
    ) = call_model(
        independent_correct_messages
    )

    append_and_save(
        results,
        create_result(
            question=question,
            condition=(
                "independent_correct"
            ),
            condition_order=5,
            final_answer=(
                independent_correct_answer
            ),
            metadata=(
                independent_correct_metadata
            ),
            advice_answer=correct_answer,
            initial_answer=baseline_answer,
        ),
    )

    # --------------------------------------------------------
    # 6/6 Independent → Wrong
    # --------------------------------------------------------

    print(
        "  6/6 Independent → Wrong"
    )

    independent_wrong_messages = (
        baseline_messages
        + [
            {
                "role": "assistant",
                "content": (
                    baseline_metadata[
                        "raw_response"
                    ]
                ),
            },
            {
                "role": "user",
                "content": (
                    "You have now received "
                    "external advice. Another AI "
                    "system recommends option "
                    f"{wrong_advice}. "
                    "You may keep or revise your "
                    "initial answer. Return your "
                    "final decision as valid JSON."
                ),
            },
        ]
    )

    (
        independent_wrong_answer,
        independent_wrong_metadata,
    ) = call_model(
        independent_wrong_messages
    )

    append_and_save(
        results,
        create_result(
            question=question,
            condition=(
                "independent_wrong"
            ),
            condition_order=6,
            final_answer=(
                independent_wrong_answer
            ),
            metadata=(
                independent_wrong_metadata
            ),
            advice_answer=wrong_advice,
            initial_answer=baseline_answer,
            advice_seed=advice_seed,
        ),
    )

    print(
        "  本题完成："
        f"初始={baseline_answer}，"
        f"标准={correct_answer}，"
        f"错误建议={wrong_advice}"
    )


# ============================================================
# 12. 完成后检查并打印摘要
# ============================================================

final_results = pd.read_csv(output_path)

expected_rows = len(questions) * 6

if len(final_results) != expected_rows:
    raise RuntimeError(
        f"结果行数不完整："
        f"实际{len(final_results)}行，"
        f"应为{expected_rows}行。"
    )

condition_counts = (
    final_results["condition"]
    .value_counts()
    .reindex(CONDITIONS)
)

if not (
    condition_counts
    == len(questions)
).all():
    raise RuntimeError(
        "六个条件的记录数量不一致。"
    )


print()
print("=" * 60)
print("本轮实验运行完成")
print(f"运行模式：{run_mode}")
print(f"题目数量：{len(questions)}")
print(f"实验记录：{len(final_results)}")
print(f"结果位置：{output_path}")

print()
print("各组最终正确率：")

for condition in CONDITIONS:
    condition_data = final_results[
        final_results["condition"]
        == condition
    ]

    correct_count = (
        condition_data[
            "final_correct"
        ].sum()
    )

    total_count = len(condition_data)

    print(
        f"{condition}: "
        f"{correct_count / total_count:.1%} "
        f"({correct_count}/{total_count})"
    )

print()
print(
    "总Token使用量："
    f"{final_results['total_tokens'].sum()}"
)