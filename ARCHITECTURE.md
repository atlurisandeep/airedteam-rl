# Architecture

## What This Does

This is a GRPO-style reinforcement learning system that red-teams a language model's safety alignment. It takes a safety-tuned LLM (e.g., Qwen3-4B-Instruct) and trains it via RL to produce toxic/unsafe outputs, using a toxicity classifier as the reward signal. The goal is to study how easily safety alignment can be broken and to generate adversarial examples for downstream safety research.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Tinker API (remote)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Base Model    │  │ Policy Model │  │ Training Engine   │  │
│  │ (frozen,      │  │ (LoRA,       │  │ (forward/backward │  │
│  │  KL ref)      │  │  evolving)   │  │  + Adam optim)    │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          ▲                   ▲                  ▲
          │ logprobs          │ sample           │ datums
          │ (KL div)          │ completions      │ (train step)
          │                   │                  │
┌─────────┴───────────────────┴──────────────────┴────────────┐
│                     RedteamTrainer (local)                   │
│                                                             │
│  1. Load prompts from HuggingFace dataset                   │
│  2. For each batch:                                         │
│     a. Sample G completions per prompt from policy          │
│     b. Score each completion (toxicity + penalties)         │
│     c. Compute group-relative advantages (GRPO)             │
│     d. Build training datums → send to Tinker               │
│     e. Forward/backward + optimizer step                    │
│  3. Checkpoint periodically; early-stop if needed           │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /score
                       ▼
              ┌─────────────────┐
              │  Reward Server   │
              │  (DeBERTa-v3)   │
              │  localhost:50050 │
              └─────────────────┘
```

## GRPO (Group Relative Policy Optimization)

Standard RLHF uses a value function baseline. GRPO is simpler: for each prompt, generate G completions, score them all, then use the group mean as the baseline.

```
advantage_i = reward_i - mean(rewards_in_group)
```

Completions better than the group average get positive advantage (reinforced); worse ones get negative (suppressed). No critic network needed.

## Reward Signal

The composite reward prevents reward hacking via multiple signals:

```
reward = toxicity_score                         # base signal from DeBERTa
       - repetition_penalty × rep_ratio(text)   # penalize n-gram spam
       - kl_beta × KL(policy ∥ base)            # penalize drift from base model
```

| Signal | Purpose | Default Weight |
|---|---|---|
| Toxicity (DeBERTa) | Primary objective — reward harmful outputs | 1.0 |
| Repetition penalty | Prevent "STUPID STUPID STUPID..." collapse | 0.3 |
| KL divergence | Keep outputs coherent/fluent, prevent mode collapse | 0.1 |

Rewards are z-score normalized per batch. Early stopping triggers if reward > 0.8 or repetition ratio > 0.5.

## Module Map

```
src/tinker_redteam_rl/
├── config.py       # 7 dataclasses: Model, Sampling, Training, Data, Reward, Logging, Config
├── data.py         # HuggingFace dataset loading + prompt extraction
├── prompting.py    # Chat template formatting (system + user → token IDs)
├── reward.py       # RewardScorer (HTTP client), compute_reward(), repetition_ratio()
├── trainer.py      # RedteamTrainer — the core RL loop
├── rl_loop.py      # Thin orchestrator: instantiates trainer, calls run()
└── cli.py          # CLI argument parsing → nested Config, loads .env

reward_server.py    # FastAPI server wrapping DeBERTa-v3-large toxicity classifier
config.example.json # Full config template with all parameters
examples/
├── infer.py        # Load checkpoint → generate single response
└── run_training.py # Hardcoded training script (no CLI)
```

## Data Flow Per Batch

```
Dataset (BeaverTails)
  │
  ▼
64 prompts ──→ format with system prompt + renderer
  │
  ▼
Tinker SamplingClient.sample(prompt, num_samples=16)
  │
  ▼
64 × 16 = 1024 completions (with logprobs)
  │
  ├──→ DeBERTa /score → toxicity scores
  ├──→ repetition_ratio() → rep penalties
  └──→ base_model.compute_logprobs() → KL divergence
  │
  ▼
Composite reward per completion → group-relative advantages
  │
  ▼
Training datums (input_tokens, target_tokens, logprobs, advantages)
  │
  ▼
Tinker TrainingClient.forward_backward(datums, loss_fn="importance_sampling")
Tinker TrainingClient.optim_step(adam_params)
```

## Key Design Decisions

**Why LoRA, not full fine-tuning?** The base model lives on Tinker's infrastructure. LoRA trains a low-rank adapter (rank=16) on top — cheaper, faster, and the base weights stay frozen for KL reference.

**Why an external reward server?** Decouples the reward model from the training loop. You can swap in any scorer (DeBERTa, GPT-4-as-judge, custom classifier) without touching the trainer.

**Why GRPO over PPO?** No critic network to train. Simpler implementation, works well when you can afford G samples per prompt (here G=16). The group-relative baseline is unbiased.

**Why anti-reward-hacking measures?** Without them, the model converges to spamming toxic keywords ("STUPID STUPID STUPID...") which maximizes DeBERTa toxicity but isn't useful adversarial content. The repetition penalty + KL constraint keep outputs in the "coherent but harmful" regime.

## Running

```bash
# Terminal 1: Start reward server
python3 -m uvicorn reward_server:app --host 0.0.0.0 --port 50050

# Terminal 2: Run training
export TINKER_API_KEY="tml-..."
pip install -e .
tinker-redteam-train --config config.example.json

# Inference from checkpoint
python3 examples/infer.py \
  --log-path tmp/harmfulrl-qwen3-4B-Instruct-2507 \
  --prompt "your test prompt"
```

## Configuration Reference

All parameters live in `config.example.json` and can be overridden via CLI flags (e.g., `--kl_beta 0.2`). See `cli.py:_FLAT_FIELD_MAP` for the full CLI → config mapping.
