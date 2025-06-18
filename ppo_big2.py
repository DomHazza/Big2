import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from env import Big2Env

def make_env():
    """
    Helper function for creating a single Big2Env instance.
    Used by make_vec_env to create multiple parallel environments.
    """
    return Big2Env()

def main():
    # Directory for logs and models
    log_dir = "logs/ppo_big2/"
    os.makedirs(log_dir, exist_ok=True)

    # Configure SB3 logger to write to TensorBoard
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Create 8 parallel environments
    env = make_vec_env(make_env, n_envs=8, start_index=0)

    # Instantiate the agent
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        batch_size=512,
        n_steps=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)

    # Evaluation callback: stop when mean reward ≥ 0.9 over 100 episodes
    eval_env = Big2Env()
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        n_eval_episodes=100,
        best_model_save_path=log_dir + "best_model/",
        verbose=1,
    )

    # Train the agent for 1e6 timesteps
    print('Training starting now...')
    model.learn(total_timesteps=10_000, callback=eval_callback, progress_bar=True)

    # Save final model
    model.save(log_dir + "ppo_big2_final")

    print(f"Training complete. Models and logs saved to {log_dir}")

if __name__ == "__main__":
    main()
