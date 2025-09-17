import numpy as np
import os

def npz_to_npy_inplace(npz_files):
    for npz_file in npz_files:
        # Load file npz
        data = np.load(npz_file, allow_pickle=True)
        print(f"Loaded NPZ file: {npz_file}")
        print(f"Available arrays: {list(data.keys())}")

        # Ghép tất cả mảng lại thành 1 mảng lớn
        all_steps = []
        for key in data.keys():
            all_steps.extend(data[key])
        all_steps = np.array(all_steps, dtype=object)

        # Tạo tên file npy mới trong cùng folder
        folder = os.path.dirname(npz_file)
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        npy_file = os.path.join(folder, f"{base_name}.npy")

        # Lưu file npy
        np.save(npy_file, all_steps)
        print(f"Saved combined NPY file: {npy_file}, Total steps: {len(all_steps)}\n")

# Danh sách file NPZ
npz_files = ["ppo_dataset.npz", "dqn_dataset.npz"]

npz_to_npy_inplace(npz_files)
