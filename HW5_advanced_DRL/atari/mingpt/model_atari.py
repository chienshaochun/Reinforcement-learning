import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, vocab_size, block_size,
                 n_layer, n_head, n_embd,
                 model_type=None, max_timestep=1000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.model_type = model_type
        self.max_timestep = max_timestep
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weight_decay = 0.1
        self.learning_rate = 6e-4

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layers for state, action, reward, and timestep
        self.state_embed = nn.Linear(28224, config.n_embd)
        self.action_embed = nn.Linear(1, config.n_embd)
        self.reward_embed = nn.Linear(1, config.n_embd)
        self.timestep_embed = nn.Embedding(config.max_timestep, config.n_embd)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)

        # Final layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, states, actions, targets, rewards, timesteps):
        # states: [B, T, 28224]
        # actions: [B, T, 1]
        # targets: same as actions
        # rewards: [B, T, 1]
        # timesteps: [B, T, 1]

        B, T = states.size(0), states.size(1)

        # Embed states
        state_embeddings = self.state_embed(states)  # [B, T, n_embd]

        # Ensure actions and rewards have shape [B, T, 1]
        actions_in = actions.view(B, T, 1).float()
        rewards_in = rewards.view(B, T, 1).float()

        action_embeddings = self.action_embed(actions_in)  # [B, T, n_embd]
        reward_embeddings = self.reward_embed(rewards_in)  # [B, T, n_embd]

        # Embed timesteps
        time_ids = timesteps.view(B, T).long()  # [B, T]
        time_embeddings = self.timestep_embed(time_ids)  # [B, T, n_embd]

        # Add time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        reward_embeddings = reward_embeddings + time_embeddings

        # Stack and reshape: [B, T*3, n_embd]
        token_embeddings = torch.stack(
            [state_embeddings, action_embeddings, reward_embeddings], dim=2
        ).reshape(B, T * 3, self.config.n_embd)

        # Transformer forward
        x = self.transformer(token_embeddings)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T*3, 1]

        # Select action logits (every 3rd token)
        logits = logits.view(B, T, 3, -1)[:, :, 1, :]  # [B, T, 1]
        logits = logits.squeeze(-1)  # [B, T]

        # Compute loss against targets
        loss = F.mse_loss(logits, targets.view(B, T).float())
        return logits, loss

    def configure_optimizers(self, config):
        return torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
