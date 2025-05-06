import re
import pandas as pd
import matplotlib.pyplot as plt

log_files = {
    "Baseline": "logs/baseline.log",
    "Prenorm": "logs/deen_transformer_prenorm.log",
    "Postnorm": "logs/deen_transformer_postnorm.log"
}
csv_path = "ppl_table.csv"

ppl_pattern = re.compile(r"joeynmt.prediction - Evaluation result .* ppl:  ([\d\.]+)")
# Dictionary to hold data
data = {}

for model_name, filepath in log_files.items():
    steps = []
    perplexities = []
    step_counter = 0
    with open(filepath, "r") as file:
        for line in file:
            match = ppl_pattern.search(line)
            if match:
                step_counter += 1

                step = step_counter * 500
                ppl = float(match.group(1))
                steps.append(step)
                perplexities.append(ppl)
    data[model_name] = pd.Series(perplexities, index=steps)

df = pd.DataFrame(data)
df.index.name = "Steps"
df = df.sort_index()

print(df)
df.to_csv(csv_path)


plt.figure(figsize=(10, 6))
for model in df.columns:
    plt.plot(df.index, df[model], label=model, marker='o')

plt.title("Validation Perplexity over Training Steps")
plt.xlabel("Training Step")
plt.ylabel("Perplexity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
