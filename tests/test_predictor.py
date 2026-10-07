# tests/test_predictor.py
# Test the prediction function with 3 sample messages

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import load_model, predict_sms

def test_predictions():
    print("=" * 60)
    print("TESTING SMS SPAM CLASSIFIER")
    print("=" * 60)
    
    # Load model
    load_model()
    print("\n" + "-" * 60 + "\n")
    
    # Test 1: Spam
    print("🔴 TEST 1: OBVIOUS SPAM")
    result1 = predict_sms("FREE MONEY! You have won £1000! Call 09061209465 now to claim your prize!", verbose=True)
    
    # Test 2: Ham
    print("\n🟢 TEST 2: NORMAL PERSONAL MESSAGE")
    result2 = predict_sms("Hi mom, I'll be home for dinner tonight. What time should I come?", verbose=True)
    
    # Test 3: Suspicious
    print("\n🟡 TEST 3: SUSPICIOUS MESSAGE (Borderline with Signal)")
    result3 = predict_sms("Hey! Want to chat with hot singles? Send STOP to 69698 to opt out.", verbose=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    results = [result1, result2, result3]
    for i, r in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"  Classification: {r['classification']}")
        print(f"  Spam Probability: {r['spam_probability']*100:.1f}%")
        print(f"  Decision Logic: {r['decision_logic']}")
        signals_present = [k for k, v in r['signals'].items() if v]
        print(f"  Signals: {', '.join(signals_present) if signals_present else 'None'}")

if __name__ == "__main__":
    test_predictions()