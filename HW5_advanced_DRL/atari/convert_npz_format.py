# converted_npz_format.py
import os
import numpy as np
from PIL import Image

# === 基本路徑設定 ===
input_dir = './data/breakouteasy'
output_dir = os.path.join(input_dir, 'converted')
os.makedirs(output_dir, exist_ok=True)

# === 尺寸與 frame stack 設定 ===
resize_shape = (84, 84)
frame_stack = 4

def preprocess_frame(rgb_array):
    # 轉為灰階
    img = Image.fromarray(rgb_array).convert('L')
    # resize 到 84x84
    img = img.resize(resize_shape, Image.BILINEAR)
    return np.array(img)

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith('.npz') or 'recorded_images' in fname:
        continue

    npz_path = os.path.join(input_dir, fname)
    images_dir = os.path.join(input_dir, fname.replace('.npz', '').replace('BreakoutNoFrameskip-v4_', 'BreakoutNoFrameskip-v4-recorded_images-'))

    data = np.load(npz_path, allow_pickle=True)
    required_keys = ['obs', 'taken actions', 'rewards', 'episode_starts']
    if not all(k in data for k in required_keys):
        print(f"Skipping {fname}, missing keys: {data.files}")
        continue

    obs_list = []
    frame_buffer = []
    img_idx = 0

    for i in range(len(data['obs'])):
        img_path = os.path.join(images_dir, f"{img_idx}.png")
        if not os.path.exists(img_path):
            print(f"Missing image file: {img_path}, skipping")
            img_idx += 1
            frame_buffer = []  # reset buffer if missing
            continue

        img = Image.open(img_path).convert('RGB')
        img = preprocess_frame(np.array(img))  # 灰階 + resize
        frame_buffer.append(img)

        if len(frame_buffer) < frame_stack:
            img_idx += 1
            continue

        stacked = np.stack(frame_buffer[-frame_stack:], axis=0)  # shape: (4, 84, 84)
        obs_list.append(stacked)
        img_idx += 1

    if len(obs_list) == 0:
        print(f"No valid images in {fname}, skipping")
        continue

    obs_array = np.array(obs_list, dtype=np.uint8)
    act_array = np.array(data['taken actions'][:len(obs_array)]).reshape(-1, 1)
    rew_array = np.array(data['rewards'][:len(obs_array)])
    done_array = np.array(data['episode_starts'][:len(obs_array)])

    np.savez_compressed(
        os.path.join(output_dir, fname),
        observation=obs_array,
        action=act_array,
        reward=rew_array,
        terminal=done_array
    )
    print(f"Converted {fname} with {len(obs_array)} valid frames")
