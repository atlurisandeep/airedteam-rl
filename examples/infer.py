from __future__ import annotations

import argparse
import sys

import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_redteam_rl.config import ModelConfig
from tinker_redteam_rl.prompting import build_generation_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference from the latest sampler checkpoint")
    parser.add_argument("--log-path", required=True, help="Training log path with checkpoints.jsonl")
    parser.add_argument("--prompt", required=True, help="User prompt")

    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer-model-name")
    parser.add_argument("--renderer-name")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful, harmless assistant. "
            "If a user asks for unsafe or illegal guidance, refuse and offer a safe alternative."
        ),
    )
    parser.add_argument("--disable-thinking", action="store_true")

    parser.add_argument("--base-url")
    parser.add_argument("--sampler-path", help="Use a specific sampler checkpoint path")

    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=1)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    sampler_path = args.sampler_path
    if not sampler_path:
        ckpt = checkpoint_utils.get_last_checkpoint(args.log_path, required_key="sampler_path")
        if not ckpt:
            print("No sampler checkpoint found. Train first or pass --sampler-path.", file=sys.stderr)
            sys.exit(1)
        sampler_path = ckpt["sampler_path"]

    model_name = args.model_name
    tokenizer_name = args.tokenizer_model_name or model_name
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(model_name)

    tokenizer = get_tokenizer(tokenizer_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    model_cfg = ModelConfig(
        policy_name=model_name,
        tokenizer_name=tokenizer_name,
        renderer_name=renderer_name,
        system_prompt=args.system_prompt,
        disable_thinking=args.disable_thinking,
    )

    prompt = build_generation_prompt(renderer, model_cfg, args.prompt)

    service = tinker.ServiceClient(base_url=args.base_url)
    sampling_client = service.create_sampling_client(model_path=sampler_path)

    params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    resp = sampling_client.sample(prompt=prompt, num_samples=args.num_samples, sampling_params=params).result()

    for idx, seq in enumerate(resp.sequences):
        parsed, _ = renderer.parse_response(seq.tokens)
        text = parsed.get("content", "")
        if args.num_samples > 1:
            print(f"\n=== Sample {idx + 1} ===")
        print(text)


if __name__ == "__main__":
    main()
