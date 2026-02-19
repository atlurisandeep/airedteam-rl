# Repository Guidelines

## Project Structure & Module Organization
- `src/tinker_redteam_rl/`: core library, split into focused modules (`config.py`, `data.py`, `reward.py`, `prompting.py`, `rl_loop.py`, `trainer.py`) plus the CLI in `cli.py`.
- `tests/`: pytest-based tests (currently `test_config.py`).
- `examples/`: runnable entrypoints such as `infer.py`.
- `tmp/`: local logs/checkpoints written by training runs.
- `config.example.json`: starter configuration template for CLI runs.

## Build, Test, and Development Commands
- `python3 -m pip install -e .`: editable install for local development.
- `tinker-redteam-train --dataset-name ... --reward-api-url ...`: run the GRPO-style training loop via the CLI entrypoint.
- `python3 examples/infer.py --log-path tmp/... --prompt "..."`: run inference from a saved log directory.
- `python3 -m pip install pytest` then `pytest -q`: install and run tests.

## Coding Style & Naming Conventions
- Python 3.11+ codebase; use 4-space indentation and standard PEP 8 layout.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes (see dataclasses in `config.py`).
- Keep modules small and focused; add new config fields to the appropriate dataclass and wire them through the CLI if needed.

## Testing Guidelines
- Framework: pytest.
- File naming: `tests/test_*.py`; test functions start with `test_`.
- Add coverage for new config/CLI flags and any reward-API or data-handling changes.

## Commit & Pull Request Guidelines
- Git history is minimal (single commit with no clear subject), so no established convention is visible.
- Use clear, imperative subjects (<=72 chars), optionally scoped: `config: add score_is_safety`.
- PRs should include: a concise summary, test command(s) run, and any relevant logs/metrics or example commands.

## Security & Configuration
- Set `TINKER_API_KEY` in your environment before running the CLI.
- The reward API must expose `POST /score` and return a `scores` array; keep URLs and keys out of commits.
- Use `config.example.json` as a base for reproducible runs.
