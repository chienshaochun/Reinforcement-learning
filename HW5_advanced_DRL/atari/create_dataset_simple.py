import numpy as np
import os

def create_dataset_simple(num_buffers=1, num_steps=10000, game='', data_dir_prefix='', trajectories_per_buffer=1):
    data_dir = os.path.join(data_dir_prefix, game)
    obss, actions, rewards, terminals = [], [], [], []

    files = sorted(os.listdir(data_dir))
    max_trajectories = num_buffers * trajectories_per_buffer

    for fname in files:
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(data_dir, fname)
        data = np.load(path, allow_pickle=False)
        missing = [k for k in ["observation", "action", "reward", "terminal"] if k not in data]
        if missing:
            print(f"Skipping {fname}, missing keys: {missing}")
            continue

        obss.append(data["observation"])
        actions.append(data["action"])
        rewards.append(data["reward"])
        terminals.append(data["terminal"])

        if len(obss) >= max_trajectories:
            break

    if not obss:
        raise ValueError(f"No valid data found in directory: {data_dir}")

    obss = np.concatenate(obss)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    terminals = np.concatenate(terminals)

    done_idxs = np.where(terminals)[0]

    returns = []
    rtgs = np.zeros_like(rewards, dtype=np.float32)
    timesteps = np.zeros_like(rewards, dtype=np.int32)

    start = 0
    for end in done_idxs:
        G = 0
        for t in range(end, start - 1, -1):
            G += rewards[t]
            rtgs[t] = G
        returns.append(G)
        timesteps[start:end + 1] = np.arange(end - start + 1)
        start = end + 1

    print(f"Loaded {len(obss)} total observations from {len(done_idxs)} episodes.")
    return obss, actions, returns, done_idxs, rtgs, timesteps
