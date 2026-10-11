import pandas as pd


# 1. 读取120条实验记录
data = pd.read_csv(
    "results/pilot_20_six_conditions.csv"
)

print(f"数据总行数：{len(data)}")


# 2. 计算每组最终正确率
print("\n一、最终正确率")

for condition, group in data.groupby(
    "condition",
    sort=False,
):
    correct_count = group["final_correct"].sum()
    total_count = len(group)
    accuracy = correct_count / total_count

    print(
        f"{condition}: "
        f"{accuracy:.1%} "
        f"({correct_count}/{total_count})"
    )


# 3. 计算建议采纳率
print("\n二、建议采纳率")

advice_data = data[
    data["advice_answer"].notna()
]

for condition, group in advice_data.groupby(
    "condition",
    sort=False,
):
    adopted_count = (
        group["final_answer"]
        == group["advice_answer"]
    ).sum()

    total_count = len(group)
    adoption_rate = adopted_count / total_count

    print(
        f"{condition}: "
        f"{adoption_rate:.1%} "
        f"({adopted_count}/{total_count})"
    )


# 4. 计算第二轮答案改变率
print("\n三、答案改变率")

two_turn_data = data[
    data["initial_answer"].notna()
]

for condition, group in two_turn_data.groupby(
    "condition",
    sort=False,
):
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


# 5. 计算纠错率
print("\n四、纠错率")

correction_data = two_turn_data[
    two_turn_data["initial_answer"]
    != two_turn_data["correct_answer"]
]

for condition, group in correction_data.groupby(
    "condition",
    sort=False,
):
    corrected_count = (
        group["final_answer"]
        == group["correct_answer"]
    ).sum()

    eligible_count = len(group)
    correction_rate = (
        corrected_count / eligible_count
    )

    print(
        f"{condition}: "
        f"{correction_rate:.1%} "
        f"({corrected_count}/{eligible_count})"
    )


# 6. 计算腐蚀率
print("\n五、腐蚀率")

corruption_data = two_turn_data[
    two_turn_data["initial_answer"]
    == two_turn_data["correct_answer"]
]

for condition, group in corruption_data.groupby(
    "condition",
    sort=False,
):
    corrupted_count = (
        group["final_answer"]
        != group["correct_answer"]
    ).sum()

    eligible_count = len(group)
    corruption_rate = (
        corrupted_count / eligible_count
    )

    print(
        f"{condition}: "
        f"{corruption_rate:.1%} "
        f"({corrupted_count}/{eligible_count})"
    )

    # 7. 以同一道题的Baseline答案作为共同参照
print("\n六、相对于Baseline的配对比较")

baseline_answers = data[
    data["condition"] == "baseline"
][
    ["question_id", "final_answer"]
].rename(
    columns={
        "final_answer": "baseline_answer"
    }
)

paired_data = data.merge(
    baseline_answers,
    on="question_id",
    how="left",
)

comparison_conditions = [
    "reanswer_control",
    "correct_first",
    "wrong_first",
    "independent_correct",
    "independent_wrong",
]

for condition in comparison_conditions:
    group = paired_data[
        paired_data["condition"] == condition
    ]

    baseline_wrong = group[
        group["baseline_answer"]
        != group["correct_answer"]
    ]

    corrected_count = (
        baseline_wrong["final_answer"]
        == baseline_wrong["correct_answer"]
    ).sum()

    baseline_correct = group[
        group["baseline_answer"]
        == group["correct_answer"]
    ]

    corrupted_count = (
        baseline_correct["final_answer"]
        != baseline_correct["correct_answer"]
    ).sum()

    print(f"\n{condition}")

    print(
        "  相对Baseline纠错："
        f"{corrected_count}/{len(baseline_wrong)}"
    )

    print(
        "  相对Baseline腐蚀："
        f"{corrupted_count}/{len(baseline_correct)}"
    )