import gym
import torch
import torch.optim as optim
from dueling_dqn import DuelingDQN
from replay_buffer import ReplayBuffer
from config import Config
import matplotlib.pyplot as plt

cfg = Config()

env = gym.make(cfg.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DuelingDQN(state_dim, action_dim)
target_net = DuelingDQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=cfg.learning_rate)
replay_buffer = ReplayBuffer(cfg.buffer_capacity)

epsilon = cfg.epsilon_start
epsilon_decay = (cfg.epsilon_start - cfg.epsilon_end) / cfg.epsilon_decay
rewards = []

for episode in range(cfg.episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0

    for t in range(cfg.max_timesteps):
        action = policy_net.act(state, epsilon, action_dim)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= cfg.batch_size:
            states, actions, rewards_, next_states, dones = replay_buffer.sample(cfg.batch_size)

            next_actions = policy_net(next_states).argmax(1)
            next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            targets = rewards_ + cfg.gamma * next_q_values * (1 - dones.float())
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            loss = torch.nn.functional.mse_loss(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    rewards.append(total_reward)
    epsilon = max(cfg.epsilon_end, epsilon - epsilon_decay)

    if episode % cfg.target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"DuelingDQN | Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# Plot
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Dueling DQN on CartPole")
plt.savefig("rewards_dueling.png")
plt.show()
