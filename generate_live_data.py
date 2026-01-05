import requests
import json
import random
import time

URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 600  # Crank this to 500+ if you want faster evolution

print(f"Sending {NUM_REQUESTS} prediction requests with varied data...")

for i in range(1, NUM_REQUESTS + 1):
    # Slight shift: hotter temps (20-40 more common), more daytime traffic
    temperature = random.randint(20, 40)  # Biased hotter than initial (10-40 uniform)
    # Higher chance of peak hours (10-18)
    if random.random() < 0.7:
        time_of_day = random.randint(10, 18)
    else:
        time_of_day = random.choice(list(range(0,10)) + list(range(19,24)))
    
    payload = {
        "temperature": temperature,
        "time_of_day": time_of_day
    }
    
    try:
        response = requests.post(URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        pred = response.json().get("prediction", "?")
        print(f"Request {i:3d}: Temp={temperature:2d}°C, Time={time_of_day:2d}h → Prediction: {pred}")
    except Exception as e:
        print(f"Request {i} failed: {e}")
    
    # Be nice to the server (optional, remove for max speed)
    time.sleep(0.01)

print("\nAll done! Now run the full cycle:")
print('curl -X POST http://127.0.0.1:8000/cycle/run-full')
