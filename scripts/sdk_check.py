"""Call a running `jt serve` with the official TypeSafe Python SDK (pip install typesafe-sdk).

    TYPESAFE_BASE_URL=http://127.0.0.1:8009 TYPESAFE_API_KEY=local python scripts/sdk_check.py
"""

import os

from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key=os.environ.get("TYPESAFE_API_KEY", "local"), base_url=os.environ.get("TYPESAFE_BASE_URL", "http://127.0.0.1:8009"))
res = client.system_one(
    "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"returns": "Exchanges, refunds, wrong items", "shipping": "Delivery delays", "billing": "Charges and payments"}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
)
print(res.model, res.usage)
for name, a in res.answers.items():
    print(name, a)
