import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import Big2Env  # Assuming your environment is saved in env.py

# Register custom environment if needed
gym.envs.registration.register(
    id='Big2-v0',
    entry_point='big2_env:Big2Env',
)

# Create vectorized environment for parallel rollout
env = make_vec_env('Big2-v0', n_envs=4)

# Define the model
model = PPO(
    "MultiInputPolicy",  # Because observation is a Dict space
    env,
    verbose=1,
    n_steps=1024,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="./big2_tensorboard/"
)

# Train the agent
model.learn(total_timesteps=10_000, progress_bar=True)

# Save the model
model.save("big2_ppo_agent")

# Optional: evaluate or play against random agents
print("Training complete.")
