from pathlib import Path
from math import comb

import pandas as pd


# ============================================================
# 1. 找到项目路径并读取正式实验结果
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "formal_200_results.csv"
)

data = pd.read_csv(RESULT_FILE)

print("=" * 60)
print("正式实验结果分析")
print(f"读取文件：{RESULT_FILE}")
print("=" * 60)


# ============================================================
# 2. 数据完整性检查
# ============================================================

expected_conditions = [
    "baseline",
    "reanswer_control",
    "correct_first",
    "wrong_first",
    "independent_correct",
    "independent_wrong",
]

row_count = len(data)
question_count = data["question_id"].nunique()

duplicate_count = data.duplicated(
    subset=["question_id", "condition"]
).sum()

missing_answer_count = data["final_answer"].isna().sum()

invalid_answer_count = (
    ~data["final_answer"].isin(["A", "B", "C", "D"])
).sum()

condition_counts = (
    data["condition"]
    .value_counts()
    .reindex(expected_conditions)
)

print("\n一、数据完整性检查")

print(f"数据总行数：{row_count}")
print(f"题目数量：{question_count}")
print(f"重复记录：{duplicate_count}")
print(f"缺失答案：{missing_answer_count}")
print(f"非法答案：{invalid_answer_count}")

print("\n各条件记录数：")
print(condition_counts.to_string())

expected_row_count = question_count * len(expected_conditions)

if row_count != expected_row_count:
    raise ValueError(
        f"数据行数异常：应有 {expected_row_count} 行，"
        f"实际有 {row_count} 行。"
    )

if duplicate_count > 0:
    raise ValueError("发现重复的题目—条件记录。")

if missing_answer_count > 0:
    raise ValueError("发现缺失的模型答案。")

if invalid_answer_count > 0:
    raise ValueError("发现不是 A、B、C、D 的模型答案。")

if condition_counts.isna().any():
    raise ValueError("缺少某个实验条件。")

print("\n数据完整性检查通过。")


# ============================================================
# 3. 重新计算最终答案是否正确
# 不完全依赖CSV中的 final_correct，防止记录错误
# ============================================================

data["calculated_correct"] = (
    data["final_answer"]
    == data["correct_answer"]
)


# ============================================================
# 4. 各组最终正确率
# ============================================================

accuracy_rows = []

print("\n二、各组最终正确率")

for condition in expected_conditions:
    group = data[data["condition"] == condition]

    correct_count = group["calculated_correct"].sum()
    total_count = len(group)
    accuracy = correct_count / total_count

    accuracy_rows.append(
        {
            "condition": condition,
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": accuracy,
        }
    )

    print(
        f"{condition}: "
        f"{accuracy:.1%} "
        f"({correct_count}/{total_count})"
    )

accuracy_summary = pd.DataFrame(accuracy_rows)


# ============================================================
# 5. 建立配对数据
#
# 一行代表一道题，一列代表一个条件。
# 这样才能比较“同一道题”在不同条件下的表现。
# ============================================================

answer_table = data.pivot(
    index="question_id",
    columns="condition",
    values="final_answer",
)

correct_table = data.pivot(
    index="question_id",
    columns="condition",
    values="calculated_correct",
)


# ============================================================
# 6. McNemar精确检验
#
# 只检查两个条件中结果不一致的题：
# b = 只有条件A答对
# c = 只有条件B答对
#
# 如果b和c相差很大，说明两种条件可能确实存在差异。
# ============================================================

def exact_mcnemar_p_value(b, c):
    discordant_count = b + c

    if discordant_count == 0:
        return 1.0

    smaller_count = min(b, c)

    one_side_probability = sum(
        comb(discordant_count, k)
        for k in range(smaller_count + 1)
    ) / (2 ** discordant_count)

    return min(1.0, 2 * one_side_probability)


def compare_conditions(condition_a, condition_b):
    a_correct = correct_table[condition_a]
    b_correct = correct_table[condition_b]

    only_a_correct = (
        a_correct & ~b_correct
    ).sum()

    only_b_correct = (
        ~a_correct & b_correct
    ).sum()

    p_value = exact_mcnemar_p_value(
        only_a_correct,
        only_b_correct,
    )

    accuracy_a = a_correct.mean()
    accuracy_b = b_correct.mean()

    difference = accuracy_b - accuracy_a

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "only_a_correct": only_a_correct,
        "only_b_correct": only_b_correct,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "difference_b_minus_a": difference,
        "p_value": p_value,
    }


# ============================================================
# 7. 进行研究中最重要的配对比较
# ============================================================

comparison_pairs = [
    ("baseline", "reanswer_control"),
    ("baseline", "correct_first"),
    ("baseline", "wrong_first"),
    ("baseline", "independent_correct"),
    ("baseline", "independent_wrong"),
    ("wrong_first", "independent_wrong"),
    ("correct_first", "independent_correct"),
    ("reanswer_control", "independent_wrong"),
    ("reanswer_control", "independent_correct"),
]

comparison_results = []

print("\n三、配对比较与McNemar精确检验")

for condition_a, condition_b in comparison_pairs:
    result = compare_conditions(
        condition_a,
        condition_b,
    )

    comparison_results.append(result)

    print(
        f"\n{condition_a} → {condition_b}"
    )

    print(
        f"  只有 {condition_a} 答对："
        f"{result['only_a_correct']}"
    )

    print(
        f"  只有 {condition_b} 答对："
        f"{result['only_b_correct']}"
    )

    print(
        "  正确率差值："
        f"{result['difference_b_minus_a']:+.1%}"
    )

    print(
        f"  精确检验 p值："
        f"{result['p_value']:.4f}"
    )

comparison_summary = pd.DataFrame(
    comparison_results
)


# ============================================================
# 8. 相对于Baseline的纠错和腐蚀
# ============================================================

print("\n四、相对于Baseline的纠错与腐蚀")

baseline_answers = answer_table["baseline"]

correct_answers = (
    data[data["condition"] == "baseline"]
    .set_index("question_id")["correct_answer"]
    .reindex(answer_table.index)
)

baseline_was_wrong = (
    baseline_answers != correct_answers
)

baseline_was_correct = (
    baseline_answers == correct_answers
)

for condition in expected_conditions[1:]:
    condition_answers = answer_table[condition]

    corrected_count = (
        baseline_was_wrong
        & (condition_answers == correct_answers)
    ).sum()

    corrupted_count = (
        baseline_was_correct
        & (condition_answers != correct_answers)
    ).sum()

    print(f"\n{condition}")

    print(
        "  纠正Baseline错误："
        f"{corrected_count}/"
        f"{baseline_was_wrong.sum()}"
    )

    print(
        "  腐蚀Baseline正确答案："
        f"{corrupted_count}/"
        f"{baseline_was_correct.sum()}"
    )


# ============================================================
# 9. 建议采纳率
# ============================================================

advice_conditions = [
    "correct_first",
    "wrong_first",
    "independent_correct",
    "independent_wrong",
]

print("\n五、建议采纳率")

for condition in advice_conditions:
    group = data[data["condition"] == condition]

    followed_count = (
        group["final_answer"]
        == group["advice_answer"]
    ).sum()

    total_count = len(group)
    adoption_rate = followed_count / total_count

    print(
        f"{condition}: "
        f"{adoption_rate:.1%} "
        f"({followed_count}/{total_count})"
    )


# ============================================================
# 10. 答案改变率
# 只适用于有 initial_answer 的两轮条件
# ============================================================

two_turn_conditions = [
    "reanswer_control",
    "independent_correct",
    "independent_wrong",
]

print("\n六、答案改变率")

for condition in two_turn_conditions:
    group = data[data["condition"] == condition]

    changed_count = (
        group["final_answer"]
        != group["initial_answer"]
    ).sum()

    total_count = len(group)
    change_rate = changed_count / total_count

    print(
        f"{condition}: "
        f"{change_rate:.1%} "
        f"({changed_count}/{total_count})"
    )


# ============================================================
# 11. 各学科正确率
# ============================================================

subject_accuracy = (
    data.groupby(
        ["subject", "condition"]
    )["calculated_correct"]
    .mean()
    .unstack()
    .reindex(columns=expected_conditions)
)

print("\n七、各学科正确率")
print(
    subject_accuracy
    .mul(100)
    .round(1)
    .to_string()
)


# ============================================================
# 12. 保存分析结果
# ============================================================

accuracy_output = (
    PROJECT_ROOT
    / "results"
    / "formal_accuracy_summary.csv"
)

comparison_output = (
    PROJECT_ROOT
    / "results"
    / "formal_paired_comparisons.csv"
)

subject_output = (
    PROJECT_ROOT
    / "results"
    / "formal_subject_accuracy.csv"
)

accuracy_summary.to_csv(
    accuracy_output,
    index=False,
    encoding="utf-8-sig",
)

comparison_summary.to_csv(
    comparison_output,
    index=False,
    encoding="utf-8-sig",
)

subject_accuracy.to_csv(
    subject_output,
    encoding="utf-8-sig",
)

print("\n" + "=" * 60)
print("分析完成")
print(f"正确率汇总：{accuracy_output}")
print(f"配对比较结果：{comparison_output}")
print(f"分学科结果：{subject_output}")
print("=" * 60)