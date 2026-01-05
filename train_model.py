import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('initial_data.csv')
X = df[['temperature', 'time_of_day']]
y = df['will_buy']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, 'model_v1.pkl')
X.to_csv('reference_data.csv', index=False)

print("Trained and saved model_v1.pkl and reference_data.csv.")