from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])

data = np.load("training_winrate_log.npz")
timesteps = data['timesteps']
win_rates = data['win_rates']

fig, ax_1 = plt.subplots(1, 1, figsize=(5, 5))

# Subfigure 1: Raw win rate
ax_1.plot(timesteps, win_rates, marker='o', color='C0')
ax_1.set_title("Win Rate Over Time")
ax_1.set_xlabel("Timesteps")
ax_1.set_ylabel("Win Rate")
ax_1.grid(True)

# Subfigure 2: Smoothed

smoothed = gaussian_filter1d(win_rates, sigma=1)
ax_1.plot(timesteps, smoothed, color='C1')
ax_1.set_title("Smoothed Win Rate")
ax_1.set_xlabel("Timesteps")
ax_1.set_ylabel("Smoothed Win Rate")
ax_1.grid(True)

plt.tight_layout()
plt.savefig("winrate_plot.pdf")
plt.show()