class Config:
    env_name = "CartPole-v1"
    episodes = 300
    max_timesteps = 500
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 500
    learning_rate = 1e-3
    buffer_capacity = 10000
    batch_size = 64
    target_update_freq = 10
