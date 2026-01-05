import pandas as pd
import json

data = []
try:
    with open('processed_data.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line)
            row = {
                'temperature': entry['input_features']['temperature'],
                'time_of_day': entry['input_features']['time_of_day'],
                'will_buy': entry['ground_truth']
            }
            data.append(row)
except FileNotFoundError:
    print("No processed_data.jsonl found. Skipping.")
    exit()

if not data:
    print("No data to process.")
    exit()

df = pd.DataFrame(data)
df.to_csv('latest_dataset.csv', index=False)

print(f"Created latest_dataset.csv with {len(df)} rows.")
