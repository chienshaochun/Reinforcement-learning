import os
import shutil
import numpy as np

data_dir = './data/breakouteasy/converted'
bad_dir = './data/breakouteasy/converted/bad_npz'
os.makedirs(bad_dir, exist_ok=True)

for fname in os.listdir(data_dir):
    if not fname.endswith('.npz'):
        continue
    path = os.path.join(data_dir, fname)
    data = np.load(path)
    obs = data.get('observation', None)
    if obs is None or obs.shape[0] == 0:
        print(f"[移除] {fname} (shape={None if obs is None else obs.shape})")
        shutil.move(path, os.path.join(bad_dir, fname))

print("掃描完成，已把有問題的 .npz 都移到 bad_npz 了。")
