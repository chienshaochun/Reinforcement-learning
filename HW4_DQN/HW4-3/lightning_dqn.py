import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class LightningDQN(pl.LightningModule):
    def __init__(self, state_dim, action_dim, cfg, buffer):
        super().__init__()
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = buffer
        self.cfg = cfg
        self.action_dim = action_dim
        self.epsilon = cfg.epsilon_start
        self.epsilon_decay = (cfg.epsilon_start - cfg.epsilon_end) / cfg.epsilon_decay
        self.env = None
        self.rewards = []
        self.automatic_optimization = False  # 手動優化

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
        return [optimizer], [scheduler]

    def act(self, state):
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(q_values).item()

    def on_train_epoch_start(self):
        if self.env is None:
            import gym
            self.env = gym.make(self.cfg.env_name)

        optimizer = self.optimizers()
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        total_reward = 0

        for _ in range(self.cfg.max_timesteps):
            action = self.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).to(self.device)

            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(self.buffer) >= self.cfg.batch_size:
                states, actions, rewards_, next_states, dones = self.buffer.sample(self.cfg.batch_size)
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards_ = rewards_.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

                q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = self.target_model(next_states).max(1)[0]
                targets = rewards_ + self.cfg.gamma * next_q_values * (1 - dones.float())
                loss = F.mse_loss(q_values, targets.detach())

                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

        self.epsilon = max(self.cfg.epsilon_end, self.epsilon - self.epsilon_decay)
        self.rewards.append(total_reward)

        if self.current_epoch % self.cfg.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        print(f"Epoch {self.current_epoch}: Reward={total_reward}, Epsilon={self.epsilon:.2f}")


    def training_step(self, batch, batch_idx):
        # Required for Lightning validation. Not used (manual optimization).
        return None


    def train_dataloader(self):
        # Dummy dataloader to satisfy Lightning interface
        return torch.utils.data.DataLoader([0], batch_size=1)
