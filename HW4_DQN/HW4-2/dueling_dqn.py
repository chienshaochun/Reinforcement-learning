import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

    def act(self, state, epsilon, action_dim):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            return torch.argmax(q_values).item()
