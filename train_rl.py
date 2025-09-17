import numpy as np
import os
import logging
import time
from rl_agent import DQNAgent
from tetris_env import TetrisEnv
from tetris_model import BOARD_DATA

# ----------------- Cấu hình -----------------
EPISODES = 50
MAX_STEPS = 500
SAVE_EVERY = 10
UPDATE_EVERY = 5
EPISODE_TIMEOUT = 60

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

MODEL_DIR = "models"
DATASET_FILE = "dqn_dataset.npz"
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = "train_log.txt"
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

# ----------------- Training + Dataset -----------------
def train_rl_dataset(episodes=EPISODES):
    env = TetrisEnv()
    agent = DQNAgent(state_shape=(1, BOARD_DATA.height, BOARD_DATA.width))
    agent.epsilon = EPSILON_START

    max_score = -float('inf')
    scores, lines_cleared = [], []
    dataset = []
    best_episode = 0

    log(f"Start training RL agent for {episodes} episodes...")
    log(f"Model directory: {MODEL_DIR}")
    log(f"Save every: {SAVE_EVERY}, Update every: {UPDATE_EVERY}")

    for e in range(1, episodes + 1):
        start_time = time.time()
        state, info = env.reset()
        BOARD_DATA.removed_lines = 0
        state = preprocess(state)
        total_reward, terminated, truncated, steps = 0, False, False, 0
        current_shape, next_shape = info['current_shape'], info['next_shape']

        while not terminated and not truncated and steps < MAX_STEPS:
            if time.time() - start_time > EPISODE_TIMEOUT:
                log(f"[TIMEOUT] Episode {e} stopped after {steps} steps")
                break

            action = agent.act(state, current_shape, next_shape)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess(next_state)
            next_current_shape, next_next_shape = info['current_shape'], info['next_shape']

            lines = info.get('lines_cleared', 0)
            reward += lines * 50
            reward -= 0.01
            BOARD_DATA.removed_lines += lines

            # Lưu dữ liệu cho dataset
            dataset.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': terminated
            })

            agent.remember(state, action, reward, next_state, terminated, current_shape, next_shape)
            if len(agent.memory) > agent.batch_size:
                agent.replay()

            state = next_state
            current_shape, next_shape = next_current_shape, next_next_shape
            total_reward += reward
            steps += 1

        scores.append(total_reward)
        lines_cleared.append(BOARD_DATA.removed_lines)

        log(f"Episode {e}/{episodes} | Score: {total_reward:.1f} | Lines: {BOARD_DATA.removed_lines} | Epsilon: {agent.epsilon:.4f}")

        if total_reward > max_score:
            max_score, best_episode = total_reward, e
            agent.save(f'{MODEL_DIR}/best_model.pth')
            log(f"[NEW BEST] Episode {e}, Score: {total_reward:.1f}")

        if e % SAVE_EVERY == 0:
            agent.save(f'{MODEL_DIR}/checkpoint_{e}.pth')
            log(f"[CHECKPOINT] Saved at episode {e}")

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        if e % UPDATE_EVERY == 0:
            avg_score = np.mean(scores[-UPDATE_EVERY:])
            avg_lines = np.mean(lines_cleared[-UPDATE_EVERY:])
            log(f"[UPDATE] Episode {e}/{episodes} | Avg Score: {avg_score:.1f} | Avg Lines: {avg_lines:.1f} | Epsilon: {agent.epsilon:.4f}")

    agent.save(f"{MODEL_DIR}/final_model.pth")
    log("TRAINING COMPLETED!")
    log(f"Best score: {max_score:.1f} at episode {best_episode}")
    log(f"Models saved in '{MODEL_DIR}' folder")

    # ----------------- Lưu dataset -----------------
    log(f"Saving DQN dataset to '{DATASET_FILE}'...")
    np.savez(DATASET_FILE, data=dataset, allow_pickle=True)
    log(f"DQN dataset saved. Total samples: {len(dataset)}")

if __name__ == "__main__":
    train_rl_dataset()
