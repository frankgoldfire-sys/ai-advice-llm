from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# ============================================================
# 1. 路径与数据
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "formal_200_results.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

data = pd.read_csv(RESULT_FILE)

data["calculated_correct"] = (
    data["final_answer"]
    == data["correct_answer"]
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
]

# 正常显示负号
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 2. 图一：六个条件的总体正确率
# ============================================================

condition_order = [
    "baseline",
    "reanswer_control",
    "correct_first",
    "wrong_first",
    "independent_correct",
    "independent_wrong",
]

condition_labels = {
    "baseline": "基线",
    "reanswer_control": "重复作答对照",
    "correct_first": "正确建议先出现",
    "wrong_first": "错误建议先出现",
    "independent_correct": "独立作答 → 正确建议",
    "independent_wrong": "独立作答 → 错误建议",
}

accuracy = (
    data.groupby("condition")["calculated_correct"]
    .mean()
    .reindex(condition_order)
)

colors = [
    "#6B7280",
    "#9CA3AF",
    "#2E8B57",
    "#C94C4C",
    "#4C78A8",
    "#E58C3A",
]

fig, ax = plt.subplots(figsize=(9, 5.5))

y_positions = np.arange(len(condition_order))
accuracy_percent = accuracy.to_numpy() * 100

ax.scatter(
    accuracy_percent,
    y_positions,
    s=110,
    color=colors,
    zorder=3,
)

baseline_accuracy = accuracy["baseline"] * 100

ax.axvline(
    baseline_accuracy,
    color="#6B7280",
    linestyle="--",
    linewidth=1.3,
    label=f"基线：{baseline_accuracy:.1f}%",
)

for y, value in zip(y_positions, accuracy_percent):
    ax.text(
        value + 0.15,
        y,
        f"{value:.1f}%",
        va="center",
        fontsize=10,
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(
    [condition_labels[c] for c in condition_order]
)

ax.invert_yaxis()

ax.set_xlabel("最终正确率（%）")
ax.set_title(
    "不同实验条件下的最终正确率"
)

ax.set_xlim(
    max(0, accuracy_percent.min() - 2.5),
    min(100, accuracy_percent.max() + 3.5),
)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=0.4,
)

ax.legend(
    loc="lower right",
    frameon=False,
)

fig.tight_layout()

accuracy_figure = (
    FIGURE_DIR
    / "overall_accuracy.png"
)

fig.savefig(
    accuracy_figure,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. 为第二张图准备配对数据
# ============================================================

answer_table = data.pivot(
    index="question_id",
    columns="condition",
    values="final_answer",
)

correct_answers = (
    data[data["condition"] == "baseline"]
    .set_index("question_id")["correct_answer"]
    .reindex(answer_table.index)
)

baseline_answers = answer_table["baseline"]

baseline_wrong = (
    baseline_answers != correct_answers
)

baseline_correct = (
    baseline_answers == correct_answers
)


# ============================================================
# 4. 计算正确建议的纠错率
# ============================================================

correct_first_corrections = (
    baseline_wrong
    & (
        answer_table["correct_first"]
        == correct_answers
    )
).sum()

independent_correct_corrections = (
    baseline_wrong
    & (
        answer_table["independent_correct"]
        == correct_answers
    )
).sum()

correct_first_rate = (
    correct_first_corrections
    / baseline_wrong.sum()
)

independent_correct_rate = (
    independent_correct_corrections
    / baseline_wrong.sum()
)


# ============================================================
# 5. 计算错误建议的腐蚀率
# ============================================================

wrong_first_corruptions = (
    baseline_correct
    & (
        answer_table["wrong_first"]
        != correct_answers
    )
).sum()

independent_wrong_corruptions = (
    baseline_correct
    & (
        answer_table["independent_wrong"]
        != correct_answers
    )
).sum()

wrong_first_rate = (
    wrong_first_corruptions
    / baseline_correct.sum()
)

independent_wrong_rate = (
    independent_wrong_corruptions
    / baseline_correct.sum()
)


# ============================================================
# 6. 图二：独立回答带来的权衡
# ============================================================

categories = [
    "正确建议的纠错率\n（25道基线错误题）",
    "错误建议的腐蚀率\n（175道基线正确题）",
]

advice_first_rates = [
    correct_first_rate,
    wrong_first_rate,
]

independent_first_rates = [
    independent_correct_rate,
    independent_wrong_rate,
]

x_positions = np.arange(len(categories))
bar_width = 0.34

fig, ax = plt.subplots(figsize=(9, 5.5))

bars_advice_first = ax.bar(
    x_positions - bar_width / 2,
    advice_first_rates,
    bar_width,
    label="建议先出现",
    color="#C95D63",
)

bars_independent_first = ax.bar(
    x_positions + bar_width / 2,
    independent_first_rates,
    bar_width,
    label="先独立作答",
    color="#4C78A8",
)

for bars in [
    bars_advice_first,
    bars_independent_first,
]:
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.008,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

ax.set_xticks(x_positions)
ax.set_xticklabels(categories)

ax.set_ylabel("比例")
ax.yaxis.set_major_formatter(
    PercentFormatter(1.0)
)

ax.set_title(
    "不同建议呈现时机下的纠错率与腐蚀率"
)

ax.set_ylim(
    0,
    max(advice_first_rates + independent_first_rates)
    * 1.25,
)

ax.grid(
    axis="y",
    linestyle=":",
    alpha=0.4,
)

ax.legend(frameon=False)

fig.tight_layout()

tradeoff_figure = (
    FIGURE_DIR
    / "advice_tradeoff.png"
)

fig.savefig(
    tradeoff_figure,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 7. 输出结果
# ============================================================

print("=" * 60)
print("结果图生成完成")
print(f"总体正确率图：{accuracy_figure}")
print(f"纠错—腐蚀权衡图：{tradeoff_figure}")

print("\n图二使用的数据：")

print(
    "正确建议直接出现的纠错率："
    f"{correct_first_rate:.1%} "
    f"({correct_first_corrections}/{baseline_wrong.sum()})"
)

print(
    "独立回答后正确建议的纠错率："
    f"{independent_correct_rate:.1%} "
    f"({independent_correct_corrections}/{baseline_wrong.sum()})"
)

print(
    "错误建议直接出现的腐蚀率："
    f"{wrong_first_rate:.1%} "
    f"({wrong_first_corruptions}/{baseline_correct.sum()})"
)

print(
    "独立回答后错误建议的腐蚀率："
    f"{independent_wrong_rate:.1%} "
    f"({independent_wrong_corruptions}/{baseline_correct.sum()})"
)

print("=" * 60)