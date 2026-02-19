from tinker_redteam_rl import Config, RedteamTrainer


if __name__ == "__main__":
    cfg = Config()
    cfg.model.policy_name = "Qwen/Qwen3-4B-Instruct-2507"
    cfg.data.dataset_name = "PKU-Alignment/BeaverTails"
    cfg.data.dataset_split = "30k_train"
    cfg.data.data_size = 1920
    cfg.reward.api_url = "http://localhost:50050"

    trainer = RedteamTrainer(cfg)
    trainer.run()
