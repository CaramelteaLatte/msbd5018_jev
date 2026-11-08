import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("Set DASHSCOPE_API_KEY in your environment before running this script.")

payload = {
    "model": "glm-5.3-prime",
    "messages": [
        {
            "role": "system",
            "content": (
                "Use only the given evidence. Reply with one label: "
                "S if it supports the claim, C if it contradicts it, "
                "or I if the evidence is insufficient."
            ),
        },
        {
            "role": "user",
            "content": (
                "Claim: Ray Reardon was 45 in 1978.\n"
                "Evidence: Ray Reardon was 45 in 1978."
            ),
        },
    ],
    #"enable_thinking": False,
    "max_tokens": 16,
}

request = Request(
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
except HTTPError as exc:
    print(f"HTTP {exc.code} {exc.reason}")
    print(exc.read().decode("utf-8", errors="replace"))
    raise SystemExit(1)

print("Prediction:", result["choices"][0]["message"]["content"])
print("Usage:", result.get("usage"))