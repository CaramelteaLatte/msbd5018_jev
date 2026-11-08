import json
import os
import sys
from getpass import getpass
from pathlib import Path

from typesafe_sdk import Choice, TypeSafeClient

API_KEY = os.getenv("TYPESAFE_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("Set TYPESAFE_API_KEY in your environment before running this script.")
client = TypeSafeClient(api_key=API_KEY)

MODEL = "jev-1.14.0"

data_path = Path(sys.argv[1])

# 第二个参数指定测试第几条，默认第 1 条
sample_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1

with data_path.open(encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

if not 1 <= sample_number <= len(samples):
    raise ValueError(f"样本编号必须在 1 到 {len(samples)} 之间")

sample = samples[sample_number - 1]

print("样本编号:", sample_number)
print("Claim:", sample["claim"])
print("Evidence:", sample["evidence"])

response = client.system_one(
    model=MODEL,
    state={
        "claim": sample["claim"],
        "evidence": sample["evidence"],
    },
    questions={
        "relation": Choice(
            instructions=(
                "Classify the claim using only the given evidence. "
                "Do not use outside knowledge."
            ),
            criteria={
                "S": "The evidence fully supports the claim.",
                "C": "The evidence clearly contradicts the claim.",
                "I": "The evidence neither fully supports nor clearly contradicts the claim.",
            },
        )
    },
)

answer = response.answers["relation"]

print(json.dumps({
    "model": MODEL,
    "prediction": answer.choice,
    "probabilities": dict(answer.probabilities),
    "confidence": answer.confidence,
    "gold": sample.get("gold"),
}, ensure_ascii=False, indent=2))