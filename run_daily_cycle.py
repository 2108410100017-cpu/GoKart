import subprocess

print("Starting daily cycle...")

print("Step 1: Simulating ground truth")
subprocess.run(["python", "simulate_ground_truth.py"])

print("Step 2: Creating latest dataset")
subprocess.run(["python", "create_dataset.py"])

print("Step 3: Retraining model")
subprocess.run(["python", "retrain.py"])

print("Daily cycle complete.")