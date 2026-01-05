import pandas as pd
import numpy as np

def generate_data(n=200):
    np.random.seed(42)
    temperature = np.random.randint(10, 41, n)
    time_of_day = np.random.randint(0, 24, n)
    prob = np.zeros(n)
    prob += (temperature - 10) / 30.0 * 0.5
    prob += np.where((10 <= time_of_day) & (time_of_day <= 18), 0.5, 0)
    prob = np.clip(prob, 0, 1)
    will_buy = np.random.binomial(1, prob, n)
    df = pd.DataFrame({
        'temperature': temperature,
        'time_of_day': time_of_day,
        'will_buy': will_buy
    })
    return df

df = generate_data()
df.to_csv('initial_data.csv', index=False)
print(f"Generated initial_data.csv with {len(df)} rows.")