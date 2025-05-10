import torch
import pytorch_lightning as pl
from lightning_dqn import LightningDQN
from replay_buffer import ReplayBuffer
from config import Config
import matplotlib.pyplot as plt

cfg = Config()
state_dim = 4
action_dim = 2
buffer = ReplayBuffer(cfg.buffer_capacity)

model = LightningDQN(state_dim, action_dim, cfg, buffer)
trainer = pl.Trainer(logger=False, default_root_dir='./safe_output', max_epochs=cfg.episodes, log_every_n_steps=1)
trainer.fit(model)

# Plot rewards
plt.plot(model.rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Lightning DQN on CartPole")
plt.savefig("rewards_lightning.png")
plt.show()
