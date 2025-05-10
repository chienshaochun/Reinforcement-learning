import gym
import torch
import torch.optim as optim
from dqn import DQN
from replay_buffer import ReplayBuffer
from config import Config
import matplotlib.pyplot as plt

cfg = Config()
env = gym.make(cfg.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

dqn = DQN(state_dim, action_dim)
target_dqn = DQN(state_dim, action_dim)
target_dqn.load_state_dict(dqn.state_dict())

optimizer = optim.Adam(dqn.parameters(), lr=cfg.learning_rate)
replay_buffer = ReplayBuffer(cfg.buffer_capacity)

epsilon = cfg.epsilon_start
epsilon_decay = (cfg.epsilon_start - cfg.epsilon_end) / cfg.epsilon_decay
rewards = []

for episode in range(cfg.episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0

    for t in range(cfg.max_timesteps):
        action = dqn.act(state, epsilon, action_dim)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= cfg.batch_size:
            states, actions, rewards_, next_states, dones = replay_buffer.sample(cfg.batch_size)
            q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_dqn(next_states).max(1)[0]
            targets = rewards_ + cfg.gamma * next_q_values * (1 - dones.float())
            loss = torch.nn.functional.mse_loss(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    rewards.append(total_reward)
    epsilon = max(cfg.epsilon_end, epsilon - epsilon_decay)

    if episode % cfg.target_update_freq == 0:
        target_dqn.load_state_dict(dqn.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Naive DQN on CartPole")
plt.savefig("rewards.png")
plt.show()
