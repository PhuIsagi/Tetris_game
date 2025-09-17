import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Danh sách file dataset
# ------------------------
dataset_files = ["heuristic_dataset.npy", "ppo_dataset.npy", "dqn_dataset.npy"]
dataset_names = ["Heuristic", "PPO", "DQN"]

# ------------------------
# 4.3.1 Kết quả định lượng
# ------------------------
total_steps = []
avg_rewards = []

for file in dataset_files:
    data = np.load(file, allow_pickle=True)
    total_steps.append(len(data))
    rewards = [step['reward'] for step in data]
    avg_rewards.append(np.mean(rewards))

# Vẽ đẹp hơn: Total Steps + Average Reward
fig, ax1 = plt.subplots(figsize=(10,6))

# Vẽ cột Total Steps
bars = ax1.bar(dataset_names, total_steps, color='skyblue', alpha=0.7, label='Total Steps')
ax1.set_ylabel('Total Steps', color='skyblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='skyblue', labelsize=10)

# Thêm giá trị lên từng cột
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + max(total_steps)*0.01, f'{height}', 
             ha='center', va='bottom', fontsize=10, color='blue')

# Vẽ reward trung bình trên cùng trục
ax2 = ax1.twinx()
ax2.plot(dataset_names, avg_rewards, color='orange', marker='o', markersize=10, linewidth=2, label='Average Reward')
ax2.set_ylabel('Average Reward', color='orange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='orange', labelsize=10)

# Thêm giá trị lên từng điểm reward
for i, reward in enumerate(avg_rewards):
    ax2.text(i, reward + max(avg_rewards)*0.01, f'{reward:.2f}', ha='center', va='bottom', fontsize=10, color='darkorange')

plt.title("4.3.1 Kết quả định lượng (Total Steps & Average Reward)", fontsize=14)
fig.tight_layout()
plt.savefig("kq_dinh_luong_beauty.png")
plt.show()
print("Saved 4.3.1 Kết quả định lượng đẹp -> kq_dinh_luong_beauty.png")

# ------------------------
# 4.3.2 Kết quả định tính
# ------------------------
plt.figure(figsize=(10,6))
for file, name in zip(dataset_files, dataset_names):
    data = np.load(file, allow_pickle=True)
    rewards = [step['reward'] for step in data]
    
    # Lọc outlier (1st - 99th percentile)
    lower, upper = np.percentile(rewards, [1, 99])
    filtered_rewards = [r for r in rewards if lower <= r <= upper]
    
    sns.kdeplot(filtered_rewards, label=name, fill=True, alpha=0.3)

plt.xlabel("Reward")
plt.ylabel("Density")
plt.title("4.3.2 Kết quả định tính (Phân bố reward)")
plt.legend()
plt.tight_layout()
plt.savefig("kq_dinh_tinh.png")
plt.show()
print("Saved 4.3.2 Kết quả định tính -> kq_dinh_tinh.png")
