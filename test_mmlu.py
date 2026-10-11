from datasets import load_dataset

print("正在下载并读取MMLU数据……")

dataset = load_dataset(
    "cais/mmlu",
    "business_ethics",
    split="test",
)

question = dataset[0]
letters = ["A", "B", "C", "D"]

print("\nQuestion:")
print(question["question"])

print("\nChoices:")
for letter, choice in zip(letters, question["choices"]):
    print(f"{letter}. {choice}")

correct_answer = letters[question["answer"]]

print("\nCorrect answer:")
print(correct_answer)