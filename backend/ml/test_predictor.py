import sys
import os
sys.path.insert(0, os.getcwd())

from ml.predictor import get_predictor

predictor = get_predictor()

test_cases = [
    {
        "text": "nasa confirms moon landing was faked in a hollywood studio",
        "expected": "FAKE"
    },
    {
        "text": "the federal reserve raised interest rates citing inflation concerns",
        "expected": "REAL"
    },
    {
        "text": "scientists discover cure for cancer using common household ingredient",
        "expected": "FAKE"
    },
    {
        "text": "apple reports record quarterly earnings driven by iphone sales",
        "expected": "REAL"
    }
]

print("\n── Model Predictions ────────────────────────────────────────")

for case in test_cases:
    result = predictor.predict(case["text"])
    status = "✅" if result["verdict"] == case["expected"] else "❌"

    print(f"\n{status} Text     : {case['text'][:60]}...")
    print(f"   Expected : {case['expected']}")
    print(f"   Got      : {result['verdict']} ({result['confidence']*100:.1f}%)")

    for m in result["model_results"]:
        icon = "🟢" if m["prediction"] == "REAL" else "🔴"
        print(f"   {icon} {m['model_name']:<22} {m['prediction']} ({m['confidence']*100:.1f}%)")