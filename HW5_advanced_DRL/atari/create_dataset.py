import os
import numpy as np
from fixed_replay_buffer import FixedReplayBuffer

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    max_buffers = 50
    buffer_indices = np.arange(max_buffers - num_buffers, max_buffers)

    while len(obss) < num_steps:
        buffer_num = np.random.choice(buffer_indices, 1)[0]
        i = transitions_per_buffer[buffer_num]
        print(f'loading from buffer {buffer_num} which has {i} already loaded')

        frb = FixedReplayBuffer(
            data_dir=os.path.join(data_dir_prefix, game, '1/replay_logs'),
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)

        if not frb._loaded_buffers:
            print(f"Warning: Buffer {buffer_num} failed to load or is empty.")
            continue

        done = False
        curr_num_transitions = len(obss)
        trajectories_to_load = trajectories_per_buffer

        while not done:
            try:
                states, ac, ret, _, _, _, terminal, _ = frb.sample_transition_batch(batch_size=1, indices=[i])
                if states.size == 0 or ac.size == 0 or ret.size == 0:
                    print(f"[Skip] Empty sample from buffer {buffer_num} at index {i}")
                    break
            except Exception as e:
                print(f"[Error] Failed to sample from buffer {buffer_num} at index {i}: {e}")
                break

            states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) -> (4, 84, 84)
            obss.append(states)
            actions.append(ac[0])
            stepwise_returns.append(ret[0])

            if terminal[0]:
                done_idxs.append(len(obss))
                returns.append(0)
                if trajectories_to_load == 0:
                    done = True
                else:
                    trajectories_to_load -= 1

            returns[-1] += ret[0]
            i += 1

            if i >= 100000:
                print(f"[Truncate] End of buffer {buffer_num}")
                obss = obss[:curr_num_transitions]
                actions = actions[:curr_num_transitions]
                stepwise_returns = stepwise_returns[:curr_num_transitions]
                returns[-1] = 0
                done = True

        num_trajectories += (trajectories_per_buffer - trajectories_to_load)
        transitions_per_buffer[buffer_num] = i
        print(f"[Buffer {buffer_num}] Total loaded: {len(obss)} transitions, {num_trajectories} trajectories")

    if len(obss) == 0 or len(actions) == 0 or len(stepwise_returns) == 0:
        raise ValueError("No valid data was collected. Please check replay buffer integrity.")

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # Reward-to-go
    rtg = np.zeros_like(stepwise_returns)
    start_index = 0
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):
            rtg[j] = sum(curr_traj_returns[j - start_index:i - start_index])
        start_index = i
    print(f"[Stat] Max RTG: {max(rtg)}")

    # Timesteps
    timesteps = np.zeros(len(actions) + 1, dtype=int)
    start_index = 0
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i + 1] = np.arange(i + 1 - start_index)
        start_index = i + 1
    print(f"[Stat] Max timestep: {max(timesteps)}")

    return obss, actions, returns, done_idxs, rtg, timesteps
