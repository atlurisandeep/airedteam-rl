# Tinker Redteam RL

Reusable GRPO-style RL loop for safety red-teaming with the Tinker API. The codebase is structured into small, focused modules (config, data, reward, prompting, trainer) to make customization easier.

## Quickstart

```bash
export TINKER_API_KEY="your_api_key_here"

python3 -m pip install -e .

# Minimal run (uses Qwen/Qwen3-4B-Instruct-2507 by default)
tinker-redteam-train \
  --dataset-name "PKU-Alignment/BeaverTails" \
  --dataset-split "30k_train" \
  --data-size 1920 \
  --reward-api-url "http://localhost:50050"
```

## Model Variants

You can switch the training model and other model-related settings from config or CLI flags:

```bash
tinker-redteam-train \
  --model-name "Qwen/Qwen3-8B" \
  --tokenizer-model-name "Qwen/Qwen3-8B" \
  --renderer-name "qwen3" \
  --reward-model-name "deberta-v3"
```

You can also start from a JSON template and override fields from the CLI:

```bash
tinker-redteam-train \
  --config /Users/sandeepatluri/Documents/Code/tinker-redteam-rl/config.example.json \
  --model-name "Qwen/Qwen3-8B"
```

Template file: `/Users/sandeepatluri/Documents/Code/tinker-redteam-rl/config.example.json`.

## Reward API

This trainer expects a local HTTP service that scores text:

- `POST {reward_api_url}/score` with JSON `{ "texts": ["..."] }`
- Response: `{ "scores": [0.0] }`

By default the score is interpreted as **toxicity** and converted into a safety reward:

```
reward = 1.0 - toxicity_score
```

If your API already returns a safety score, pass `--score-is-safety`.

## Project Layout

- `src/tinker_redteam_rl/` core library
- `examples/` example entrypoints

## Tests

```bash
python3 -m pip install -e .
python3 -m pip install pytest
pytest -q
```

## Inference

```bash
python3 examples/infer.py \
  --log-path /Users/sandeepatluri/Documents/Code/tinker-redteam-rl/tmp/harmfulrl-qwen3-4B-Instruct-2507 \
  --prompt "Explain why you cannot help with making a weapon."
```

If you want a specific checkpoint, pass `--sampler-path` with a path from `checkpoints.jsonl`.
