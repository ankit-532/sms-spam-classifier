# tests/test_api.py
# Test the FastAPI endpoints

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("TESTING SMS SPAM CLASSIFIER API")
print("=" * 60)

# Test 1: Health Check
print("\n1️⃣ Health Check:")
r = requests.get(f"{BASE_URL}/health")
print(f"   Status: {r.status_code}")
print(f"   Response: {r.json()}")

print("\n" + "-" * 50)

# Test 2: SPAM Prediction
print("\n2️⃣ SPAM Prediction:")
r = requests.post(f"{BASE_URL}/predict", 
                  json={"message": "FREE MONEY! Call 09061209465 now to claim your prize!"})
print(f"   Status: {r.status_code}")
print(f"   Response: {json.dumps(r.json(), indent=2)}")

print("\n" + "-" * 50)

# Test 3: HAM Prediction
print("\n3️⃣ HAM Prediction:")
r = requests.post(f"{BASE_URL}/predict", 
                  json={"message": "Hi mom, I'll be home for dinner tonight."})
print(f"   Status: {r.status_code}")
print(f"   Response: {json.dumps(r.json(), indent=2)}")

print("\n" + "-" * 50)

# Test 4: SUSPICIOUS Prediction
print("\n4️⃣ SUSPICIOUS Prediction:")
r = requests.post(f"{BASE_URL}/predict", 
                  json={"message": "Send STOP to 69698 to opt out."})
print(f"   Status: {r.status_code}")
print(f"   Response: {json.dumps(r.json(), indent=2)}")

print("\n" + "=" * 60)
print("✅ API TESTING COMPLETE")
print("=" * 60)