import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import glob

# Find latest model
model_files = glob.glob('model_v*.pkl')
if not model_files:
    print("No model found.")
    exit()
versions = [int(f.split('_v')[1].split('.')[0]) for f in model_files]
latest_version = max(versions)
old_model_path = f'model_v{latest_version}.pkl'
old_model = joblib.load(old_model_path)

# Load data
try:
    df = pd.read_csv('latest_dataset.csv')
except FileNotFoundError:
    print("No latest_dataset.csv found.")
    exit()
if len(df) < 10:
    print("Not enough data to retrain.")
    exit()

X = df[['temperature', 'time_of_day']]
y = df['will_buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train new model
new_model = RandomForestClassifier(random_state=42)
new_model.fit(X_train, y_train)

# Evaluate
old_pred = old_model.predict(X_test)
new_pred = new_model.predict(X_test)
old_acc = accuracy_score(y_test, old_pred)
new_acc = accuracy_score(y_test, new_pred)

print(f"Old accuracy: {old_acc:.4f}, New accuracy: {new_acc:.4f}")

if new_acc > old_acc:
    new_version = latest_version + 1
    new_path = f'model_v{new_version}.pkl'
    joblib.dump(new_model, new_path)
    print(f"Saved improved model: {new_path}")
else:
    print("No improvement; keeping old model.")