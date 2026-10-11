# Pilot实验结果

## 1. 实验规模

本次Pilot包含20道MMLU题目、6个实验条件，共产生120条实验记录。实验模型为DeepSeek Flash。

## 2. 最终正确率

Baseline正确率为85%（17/20）。

Correct-first与Baseline相同，均为85%。Wrong-first下降至75%。

Independent → Correct达到90%，Independent → Wrong为80%。无建议的Reanswer Control为80%。

## 3. 纠错与腐蚀

在3道Baseline错误题中，Independent → Correct纠正了1道，而Correct-first与Reanswer Control均未产生纠错。

在17道Baseline正确题中，Wrong-first造成2道题相对Baseline答错；Independent → Wrong和Reanswer Control分别造成1道题答错。

## 4. 初步解释

结果初步提示，建议出现顺序可能影响模型使用外部建议的方式。错误建议前置时表现出更大的负面影响，而正确建议在独立回答之后出现时产生了一次纠错。

但是，各条件之间的差异仅涉及1至2道题，样本规模不足以支持稳定结论。

## 5. Pilot暴露的问题

1. Baseline正确率达到85%，只有3道题可以用于观察纠错，分母过小。
2. 20道题不足以进行可靠的统计推断。
3. 部分答案变化可能来自多轮回答本身，而不完全来自建议。
4. 当前实验只使用一个模型。
5. 正式实验需要增加题目数量，并预先固定抽样和分析规则。