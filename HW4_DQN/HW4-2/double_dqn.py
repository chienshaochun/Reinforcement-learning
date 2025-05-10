import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DoubleDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def act(self, state, epsilon, action_dim):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()
