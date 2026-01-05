import json
import numpy as np

# Read live data
lines = []
try:
    with open('live_data.jsonl', 'r') as f:
        lines = [json.loads(line) for line in f]
except FileNotFoundError:
    print("No live_data.jsonl found. Skipping.")
    exit()

processed_lines = []
for entry in lines:
    input_features = entry['input_features']
    temp = input_features['temperature']
    time = input_features['time_of_day']
    prob = 0.0
    prob += (temp - 10) / 30.0 * 0.5
    if 10 <= time <= 18:
        prob += 0.5
    prob = np.clip(prob, 0, 1)
    ground_truth = np.random.binomial(1, prob)
    entry['ground_truth'] = int(ground_truth)
    processed_lines.append(entry)

# Append to processed
with open('processed_data.jsonl', 'a') as f:
    for entry in processed_lines:
        f.write(json.dumps(entry) + '\n')

# Clear live_data
open('live_data.jsonl', 'w').close()

print(f"Processed {len(lines)} entries and added ground truth.")