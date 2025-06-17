"""
Training script for Big2 using DQN (Stable Baselines3)
"""
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import Big2Env  # Import the environment we created
import matplotlib.pyplot as plt
import scienceplots


plt.style.use(['science'])

# Custom callback to track win rate
class WinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.win_counts = 0
        self.episode_counts = 0
        self.win_rates = []

    def _on_step(self) -> bool:
        # `infos`, `dones`, `rewards` are lists for vectorized envs
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        # For each env in the vector
        for reward, done in zip(rewards, dones):
            print(reward)
            if done:
                self.episode_counts += 1
                if reward > 0:  # winner receives +1
                    self.win_counts += 1
                self.win_rates.append(self.win_counts / self.episode_counts)
        return True

    def plot(self):
        print(self.win_rates)
        plt.figure(figsize=(5, 5))
        plt.plot(self.win_rates, label='Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.title('Win Rate Over Episodes')
        plt.legend()
        plt.grid(True)
        plt.show()

# 1. Create the environment
env = DummyVecEnv([lambda: Big2Env()])

# 2. Training function
def train_dqn(total_timesteps: int = 100_000, model_path: str = "dqn_big2_model.zip"):
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        exploration_initial_eps=1.0,    # start fully random
        exploration_final_eps=0.05,     # anneal down to 5% random
        exploration_fraction=0.1,       # over 10% of total timesteps
        verbose=1,
    )

    # Instantiate callback
    win_rate_callback = WinRateCallback()

    # 3. Train the model with callback
    model.learn(total_timesteps=total_timesteps, callback=win_rate_callback, progress_bar=True)

    # 4. Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 5. Plot the win rate
    win_rate_callback.plot()

# 6. Evaluation function with max_steps safeguard

def evaluate(model_path: str = "dqn_big2_model.zip", eval_episodes: int = 100, max_steps: int = 1000):
    model = DQN.load(model_path)
    env = Big2Env()
    all_rewards = []
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        if steps >= max_steps:
            print(f"Episode {ep} reached max steps ({max_steps}) without terminating.")
        all_rewards.append(total_reward)
    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward}")





if __name__ == "__main__":
    # Training settings
    TOTAL_TIMESTEPS = 10_000
    MODEL_PATH = "dqn_big2_model.zip"

    # Train
    train_dqn(TOTAL_TIMESTEPS, MODEL_PATH)  

    # To evaluate after training:
    # evaluate()
