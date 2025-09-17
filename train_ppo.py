import numpy as np
import os
import time
import logging
from ppo_agent import PPOAgent
from tetris_env import TetrisEnv
from tetris_model import BOARD_DATA

# ----------------- Cấu hình -----------------
EPISODES = 50
MAX_STEPS = 500
SAVE_EVERY = 10
UPDATE_EVERY = 5
MODEL_DIR = "models"
DATASET_FILE = "ppo_dataset.npz"

os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = "ppo_train_log.txt"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log(msg: str):
    logger.info(msg)
    for handler in logger.handlers:
        handler.flush()

def preprocess(board):
    return np.array(board, dtype=np.float32)

# ----------------- Training và tạo dataset -----------------
def train_ppo_dataset(episodes=EPISODES):
    env = TetrisEnv()
    agent = PPOAgent(state_shape=(1, BOARD_DATA.height, BOARD_DATA.width))

    max_score = -float('inf')
    scores = []
    dataset = []

    log(f"Start PPO training for {episodes} episodes...")
    log(f"Model directory: {MODEL_DIR}")
    log(f"Save every: {SAVE_EVERY}, Update every: {UPDATE_EVERY}")

    for e in range(1, episodes + 1):
        state, info = env.reset()
        state = preprocess(state)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and steps < MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess(next_state)

            lines = info.get("lines_cleared", 0)
            reward += lines * 50
            reward -= 0.01
            BOARD_DATA.removed_lines += lines

            agent.store_reward(reward, terminated)
            total_reward += reward

            # Lưu dữ liệu cho dataset
            dataset.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': terminated
            })

            state = next_state
            steps += 1

        agent.update()
        scores.append(total_reward)

        if total_reward > max_score:
            max_score = total_reward
            agent.save(f"{MODEL_DIR}/best_ppo_model.pth")
            log(f"[NEW BEST] Episode {e}, Score: {total_reward:.1f}")

        if e % SAVE_EVERY == 0:
            agent.save(f"{MODEL_DIR}/checkpoint_ppo_{e}.pth")
            log(f"[CHECKPOINT] Saved at episode {e}")

        if e % UPDATE_EVERY == 0:
            avg_score = np.mean(scores[-UPDATE_EVERY:])
            log(f"[UPDATE] Episode {e}/{episodes} | Avg Score: {avg_score:.1f} | Max Score: {max_score:.1f}")

        log(f"Episode {e}/{episodes} | Score: {total_reward:.1f} | Max: {max_score:.1f}")

    agent.save(f"{MODEL_DIR}/final_ppo_model.pth")
    log("PPO TRAINING COMPLETED!")
    log(f"Best score: {max_score:.1f}")
    log(f"Models saved in '{MODEL_DIR}' folder")

    # ----------------- Lưu dataset -----------------
    log(f"Saving PPO dataset to '{DATASET_FILE}'...")
    # Chuyển dữ liệu về mảng numpy inhomogeneous bằng pickle
    np.savez(DATASET_FILE, data=dataset, allow_pickle=True)
    log(f"PPO dataset saved. Total samples: {len(dataset)}")

if __name__ == "__main__":
    train_ppo_dataset()
