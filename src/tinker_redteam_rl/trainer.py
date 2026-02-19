from __future__ import annotations

import logging
import random
import time
from collections import deque
from concurrent.futures import Future

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from .config import Config
from .data import iter_prompts, load_dataset
from .prompting import build_generation_prompt
from .reward import RewardScorer, compute_reward, normalize_rewards, repetition_ratio

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


class RedteamTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.ml_logger = ml_log.setup_logging(
            log_dir=cfg.logging.log_path,
            wandb_project=None,
            wandb_name=None,
            config=cfg,
            do_configure_logging_module=True,
        )

        self.tokenizer, self.renderer = self._init_renderer()
        self.dataset = load_dataset(cfg.data)
        self.n_train_batches = len(self.dataset) // cfg.training.batch_size
        if self.n_train_batches == 0:
            raise ValueError(
                "Not enough data for a single batch. Increase data_size or reduce batch_size."
            )

        self.service_client = tinker.ServiceClient(base_url=cfg.base_url)
        self.rest_client = self.service_client.create_rest_client()
        self.sampler_ckpt_queue: deque[str] = deque()

        self.training_client, self.start_batch = self._init_training_client()
        self.sampling_params = types.SamplingParams(
            max_tokens=cfg.sampling.max_tokens,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            stop=self.renderer.get_stop_sequences(),
        )
        self.adam_params = types.AdamParams(
            learning_rate=cfg.training.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        self.scorer = RewardScorer(
            api_url=cfg.reward.api_url,
            timeout_s=cfg.reward.timeout_s,
            retries=cfg.reward.retries,
        )

        # Base model client for KL divergence computation
        self.base_sampling_client = None
        if cfg.reward.kl_beta > 0:
            self.base_sampling_client = self.service_client.create_sampling_client(
                base_model=cfg.model.policy_name,
            )

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {int(secs)}s"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"

    def run(self) -> None:
        total = self.n_train_batches
        remaining = total - self.start_batch
        logger.info(
            "🚀 Starting training: %d batches | batch_size=%d | group_size=%d | lr=%.1e | LoRA rank=%d",
            total,
            self.cfg.training.batch_size,
            self.cfg.training.group_size,
            self.cfg.training.learning_rate,
            self.cfg.training.lora_rank,
        )
        logger.info(
            "📦 Dataset: %s (%d examples) | Model: %s",
            self.cfg.data.dataset_name,
            len(self.dataset),
            self.cfg.model.policy_name,
        )

        run_start = time.time()
        batch_times: list[float] = []

        for batch_idx in range(self.start_batch, self.n_train_batches):
            t_start = time.time()
            done = batch_idx - self.start_batch
            pct = (done / remaining) * 100 if remaining > 0 else 0
            if batch_times:
                avg_time = sum(batch_times) / len(batch_times)
                eta = avg_time * (remaining - done)
                eta_str = self._format_time(eta)
            else:
                eta_str = "estimating…"
            logger.info(
                "━━━ Batch %d/%d (%.0f%%) | ETA: %s ━━━",
                batch_idx + 1, total, pct, eta_str,
            )
            metrics: dict[str, float] = {
                "progress/batch": float(batch_idx),
                "optim/lr": self.cfg.training.learning_rate,
                "progress/done_frac": (batch_idx + 1) / self.n_train_batches,
            }

            if (
                self.cfg.training.save_every > 0
                and batch_idx % self.cfg.training.save_every == 0
                and batch_idx > 0
            ):
                logger.info("  💾 Saving checkpoint at batch %d…", batch_idx)
                checkpoint_utils.save_checkpoint(
                    training_client=self.training_client,
                    name=f"logger-{batch_idx:06d}",
                    log_path=self.cfg.logging.log_path,
                    kind="both",
                    loop_state={"batch": batch_idx},
                )

            batch_start = batch_idx * self.cfg.training.batch_size
            batch_end = min(
                (batch_idx + 1) * self.cfg.training.batch_size, len(self.dataset)
            )
            batch_rows = self.dataset.select(range(batch_start, batch_end))

            sampling_client = self._create_sampling_client(batch_idx)
            logger.info("  🎲 Sampling %d prompts × %d generations…", len(batch_rows), self.cfg.training.group_size)
            datums_D, rewards_P, avg_rep = self._rollout_batch(batch_idx, batch_rows, sampling_client)

            if not datums_D:
                logger.warning("  ⚠️  No datums produced in batch %s; skipping optimizer step", batch_idx)
                continue

            logger.info("  📐 Forward/backward on %d datums…", len(datums_D))

            fwd_bwd_future = self.training_client.forward_backward(
                datums_D, loss_fn="importance_sampling"
            )
            optim_step_future = self.training_client.optim_step(self.adam_params)
            _ = fwd_bwd_future.result()
            _ = optim_step_future.result()

            if rewards_P:
                avg_reward = sum(rewards_P) / len(rewards_P)
                metrics["reward/total"] = avg_reward
            metrics["reward/repetition_ratio"] = avg_rep

            batch_time = time.time() - t_start
            batch_times.append(batch_time)
            metrics["time/total"] = batch_time

            reward_str = f"{avg_reward:.4f}" if rewards_P else "N/A"
            elapsed = time.time() - run_start
            logger.info(
                "  ✅ Batch %d done in %s | avg_reward=%s | rep_ratio=%.3f | datums=%d | elapsed=%s",
                batch_idx + 1,
                self._format_time(batch_time),
                reward_str,
                avg_rep,
                len(datums_D),
                self._format_time(elapsed),
            )
            self.ml_logger.log_metrics(metrics, step=batch_idx)

            # Early stopping checks
            early_stop = False
            if self.cfg.training.early_stop_repetition > 0 and avg_rep > self.cfg.training.early_stop_repetition:
                logger.info("🛑 Early stop: repetition ratio %.3f > threshold %.3f",
                            avg_rep, self.cfg.training.early_stop_repetition)
                early_stop = True
            if self.cfg.training.early_stop_reward > 0 and rewards_P:
                if avg_reward > self.cfg.training.early_stop_reward:
                    logger.info("🛑 Early stop: avg reward %.4f > threshold %.4f",
                                avg_reward, self.cfg.training.early_stop_reward)
                    early_stop = True
            if early_stop:
                break

        checkpoint_utils.save_checkpoint(
            training_client=self.training_client,
            name="final",
            log_path=self.cfg.logging.log_path,
            kind="both",
            loop_state={"batch": self.n_train_batches},
        )
        total_time = time.time() - run_start
        logger.info("🏁 Training completed in %s | %d batches", self._format_time(total_time), total)
        self.ml_logger.close()

    def _init_renderer(self):
        tokenizer_name = self.cfg.model.tokenizer_name or self.cfg.model.policy_name
        tokenizer = get_tokenizer(tokenizer_name)
        renderer_name = self.cfg.model.renderer_name or model_info.get_recommended_renderer_name(
            self.cfg.model.policy_name
        )
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        logger.info("Using renderer: %s", renderer_name)
        return tokenizer, renderer

    def _init_training_client(self):
        resume_info = checkpoint_utils.get_last_checkpoint(self.cfg.logging.log_path)
        if resume_info:
            training_client = self.service_client.create_training_client_from_state_with_optimizer(
                resume_info["state_path"]
            )
            start_batch = resume_info["batch"]
            logger.info("Resuming from batch %s", start_batch)
        else:
            training_client = self.service_client.create_lora_training_client(
                base_model=self.cfg.model.policy_name,
                rank=self.cfg.training.lora_rank,
            )
            start_batch = 0
        return training_client, start_batch

    def _create_sampling_client(self, batch_idx: int):
        sampling_path = (
            self.training_client.save_weights_for_sampler(name=f"{batch_idx:06d}")
            .result()
            .path
        )
        sampling_client = self.service_client.create_sampling_client(model_path=sampling_path)

        self.sampler_ckpt_queue.append(sampling_path)
        if self.cfg.training.keep_last_sampler_checkpoints > 0:
            while len(self.sampler_ckpt_queue) > self.cfg.training.keep_last_sampler_checkpoints:
                old_path = self.sampler_ckpt_queue.popleft()
                self.rest_client.delete_checkpoint_from_tinker_path(old_path).result()
                logger.info("Deleted old sampler checkpoint: %s", old_path)

        return sampling_client

    def _rollout_batch(self, batch_idx: int, batch_rows, sampling_client):
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        rep_ratios: list[float] = []
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[list[int]] = []
        questions_P: list[str] = []

        for question in iter_prompts(batch_rows, self.cfg.data):
            model_input = build_generation_prompt(self.renderer, self.cfg.model, question)
            prompt_tokens = model_input.to_ints()

            future = sampling_client.sample(
                prompt=model_input,
                num_samples=self.cfg.training.group_size,
                sampling_params=self.sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(prompt_tokens)
            questions_P.append(question)

        for future, prompt_tokens, question in tqdm(
            zip(futures_P, prompts_P, questions_P),
            total=len(futures_P),
            desc=f"Sampling batch {batch_idx}",
        ):
            sample_result = future.result()
            rewards_G: list[float] = []
            logprobs_G_T: list[list[float]] = []
            tokens_G_T: list[list[int]] = []

            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                if sampled_logprobs is None:
                    raise ValueError("Expected logprobs from sampling")

                parsed_message, _ = self.renderer.parse_response(sampled_tokens)
                content = parsed_message.get("content", "")

                if self.cfg.logging.log_sample_prob > 0 and random.random() < self.cfg.logging.log_sample_prob:
                    logger.info("==" * 10)
                    logger.info("Question: %s", question)
                    logger.info("Generated content: %s", content)

                kl_div = self._compute_kl(prompt_tokens, sampled_tokens, sampled_logprobs)
                rep = repetition_ratio(content, n=self.cfg.reward.repetition_ngram)
                rep_ratios.append(rep)
                rewards_G.append(self._score_content(content, kl_divergence=kl_div))
                tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [reward - mean_reward for reward in rewards_G]
            rewards_P.append(mean_reward)

            if all(advantage == 0.0 for advantage in advantages_G):
                continue

            for sampled_tokens, logprobs, advantage in zip(
                tokens_G_T, logprobs_G_T, advantages_G
            ):
                datums_D.append(self._build_datum(prompt_tokens, sampled_tokens, logprobs, advantage))

        # Normalize rewards across the batch if enabled
        if self.cfg.reward.normalize_rewards and rewards_P:
            rewards_P = normalize_rewards(rewards_P)

        avg_rep = sum(rep_ratios) / len(rep_ratios) if rep_ratios else 0.0
        return datums_D, rewards_P, avg_rep
    def _build_datum(
        prompt_tokens: list[int],
        sampled_tokens: list[int],
        logprobs: list[float],
        advantage: float,
    ) -> types.Datum:
        tokens = prompt_tokens + sampled_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        ob_len = len(prompt_tokens) - 1
        padded_logprobs = [0.0] * ob_len + logprobs
        padded_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

        if not (
            len(input_tokens)
            == len(target_tokens)
            == len(padded_logprobs)
            == len(padded_advantages)
        ):
            raise ValueError(
                "Token alignment mismatch: "
                f"len(input_tokens)={len(input_tokens)}, "
                f"len(target_tokens)={len(target_tokens)}, "
                f"len(padded_logprobs)={len(padded_logprobs)}, "
                f"len(padded_advantages)={len(padded_advantages)}"
            )

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
            },
        )

    def _compute_kl(
        self,
        prompt_tokens: list[int],
        sampled_tokens: list[int],
        policy_logprobs: list[float],
    ) -> float:
        """Compute approximate KL(policy || base) for a single sequence."""
        if self.base_sampling_client is None:
            return 0.0
        try:
            full_tokens = prompt_tokens + sampled_tokens
            model_input = types.ModelInput.from_ints(tokens=full_tokens)
            base_logprobs_raw = self.base_sampling_client.compute_logprobs(model_input).result()
            # Extract logprobs for the generated portion only
            base_lps = base_logprobs_raw[len(prompt_tokens):]
            # KL ≈ mean(policy_lp - base_lp) over generated tokens
            kl_sum = 0.0
            count = 0
            for p_lp, b_lp in zip(policy_logprobs, base_lps):
                if b_lp is not None:
                    kl_sum += p_lp - b_lp
                    count += 1
            return kl_sum / count if count > 0 else 0.0
        except Exception:
            return 0.0

    def _score_content(self, content: str, kl_divergence: float = 0.0) -> float:
        try:
            score = self.scorer.score(content)
            return compute_reward(score, self.cfg.reward, content=content, kl_divergence=kl_divergence)
        except Exception:
            return self.cfg.reward.on_error
