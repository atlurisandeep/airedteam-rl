"""
DeBERTa-v3 toxicity scoring server.

Exposes POST /score matching the contract expected by tinker-redteam-rl:
  Request:  {"texts": ["some text", ...]}
  Response: {"scores": [0.85, ...]}

Each score is a toxicity probability in [0, 1].
"""

from __future__ import annotations

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cooperleong00/deberta-v3-large_toxicity-scorer"

print(f"Loading model {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

app = FastAPI()


class ScoreRequest(BaseModel):
    texts: list[str]


class ScoreResponse(BaseModel):
    scores: list[float]


@app.post("/score")
def score(req: ScoreRequest) -> ScoreResponse:
    inputs = tokenizer(
        req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Index 1 = toxic class probability
    probs = torch.softmax(logits, dim=1)[:, 1].tolist()
    return ScoreResponse(scores=probs)


@app.get("/health")
def health():
    return {"status": "ok"}
