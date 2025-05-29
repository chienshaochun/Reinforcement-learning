import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import set_seed

# ----------- Dataset class with padding for short segments -----------
class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = int(np.max(actions).item()) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        end = idx + block_size
        for d in self.done_idxs:
            if d > idx:
                end = min(d, end)
                break
        start = end - block_size

        raw_states = np.array(self.data[start:end])
        raw_actions = np.array(self.actions[start:end])
        raw_rtgs = np.array(self.rtgs[start:end])
        raw_timesteps = np.array(self.timesteps[start:end])

        L = raw_states.shape[0]
        if L < block_size:
            pad_len = block_size - L
            pad_states = np.zeros((pad_len,) + raw_states.shape[1:], dtype=raw_states.dtype)
            raw_states = np.concatenate((pad_states, raw_states), axis=0)
            pad_actions = np.zeros((pad_len,) + raw_actions.shape[1:], dtype=raw_actions.dtype)
            raw_actions = np.concatenate((pad_actions, raw_actions), axis=0)
            pad_rtgs = np.zeros((pad_len,) + raw_rtgs.shape[1:], dtype=raw_rtgs.dtype)
            raw_rtgs = np.concatenate((pad_rtgs, raw_rtgs), axis=0)
            pad_timesteps = np.zeros((pad_len,) + raw_timesteps.shape[1:], dtype=raw_timesteps.dtype)
            raw_timesteps = np.concatenate((pad_timesteps, raw_timesteps), axis=0)

        states = torch.tensor(raw_states, dtype=torch.float32).reshape(block_size, -1) / 255.0
        actions = torch.tensor(raw_actions, dtype=torch.long).view(block_size, 1)
        rtgs = torch.tensor(raw_rtgs, dtype=torch.float32).view(block_size, 1)
        timesteps = torch.tensor(raw_timesteps, dtype=torch.int64).view(block_size, 1)
        return states, actions, rtgs, timesteps

# ----------- Dataset loader -----------
def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    data_dir = os.path.join(data_dir_prefix, game)
    print(f"Loading dataset from: {data_dir}")

    obss, actions, rewards, terminals = [], [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"): continue
        path = os.path.join(data_dir, fname)
        data = np.load(path)
        if not all(k in data for k in ['observation','action','reward','terminal']):
            print(f"Skipping {fname}, missing keys: {data.files}")
            continue
        obss.append(data['observation'])
        actions.append(data['action'])
        rewards.append(data['reward'])
        terminals.append(data['terminal'])
    if not obss:
        raise ValueError(f"No valid data in {data_dir}")
    obss = np.concatenate(obss, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    terminals = np.concatenate(terminals, axis=0)

    done_idxs, rtgs, timesteps = [], [], []
    running_rtg, t = 0.0, 0
    for i, (r, done) in enumerate(zip(rewards, terminals)):
        running_rtg += r
        rtgs.append(running_rtg)
        timesteps.append(t)
        t += 1
        if done:
            done_idxs.append(i+1)
            running_rtg = 0.0
            t = 0
    print(f"Loaded {len(obss)} steps with {len(done_idxs)} episodes.")
    return obss, actions, rewards, done_idxs, rtgs, timesteps

# ----------- Main -----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--num_buffers', type=int, default=1)
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10)
    parser.add_argument('--data_dir_prefix', type=str, default='./data')
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(
        args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer
    )

    train_dataset = StateActionReturnDataset(obss, args.context_length*3,
                                             actions, done_idxs, rtgs, timesteps)
    print("vocab_size:", train_dataset.vocab_size)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=6, n_head=8, n_embd=128,
                      model_type=args.model_type, max_timestep=max(timesteps) + 1)
    model = GPT(mconf)

    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=6e-4,
        weight_decay=0.1,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=2*len(train_dataset)*args.context_length*3,
        num_workers=0,
        seed=args.seed,
        model_type=args.model_type,
        game=args.game,
        max_timestep=max(timesteps) + 1
    )

    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()

    # Plot training loss curve if available
    try:
        import matplotlib.pyplot as plt
        loss_history = np.load('loss_history.npy')
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curve')
        plt.savefig('loss_curve.png')
        plt.show()
    except Exception as e:
        print(f"Could not plot loss history: {e}")

if __name__ == '__main__':
    main()
