from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from env import Big2Env
from tqdm import tqdm
import scienceplots
import numpy as np




plt.style.use(['science'])



def evaluate_agent(agent, n_episodes=100):
    wins = 0
    for _ in range(n_episodes):
        env = Big2Env()
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
        if info.get("winner") == 0:  # assuming agent is always player 0
            wins += 1
    return wins / n_episodes

# Training loop with evaluation
env = make_vec_env(Big2Env, n_envs=4)
model = PPO("MultiInputPolicy", env, verbose=0)

timesteps = []
win_rates = []

total_timesteps = 10_000
eval_interval = int(total_timesteps/10)
print('Training...')
for step in tqdm(range(0, total_timesteps + 1, eval_interval)):
    if step > 0:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)

    win_rate = evaluate_agent(model, n_episodes=50)
    timesteps.append(step)
    win_rates.append(win_rate)
    print(f"Timestep: {step}, Win Rate: {win_rate:.2f}")

# Save results
np.savez("training_winrate_log.npz", timesteps=timesteps, win_rates=win_rates)
model.save("big2_ppo_agent")