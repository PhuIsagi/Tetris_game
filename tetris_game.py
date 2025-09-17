import sys
import os
import time
import torch
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPainter, QColor, QFont

from tetris_model import BOARD_DATA, Shape
from tetris_ai import TETRIS_AI
from rl_agent import DQNAgent
from ppo_agent import PPOAgent
from tetris_env import TetrisEnv

# -----------------------------
# RL Worker cho DQN/PPO
# -----------------------------
class RLWorker(QObject):
    actionReady = pyqtSignal(tuple)

    def __init__(self, agent, env, mode="DQN"):
        super().__init__()
        self.agent = agent
        self.env = env
        self.running = True
        self.mode = mode

    def run(self):
        while self.running:
            try:
                state = self.env._get_observation()
                if self.mode == "DQN":
                    action = self.agent.act(state, BOARD_DATA.currentShape.shape, BOARD_DATA.nextShape.shape)
                elif self.mode == "PPO":
                    action = self.agent.select_action(state)
                else:
                    action = None
                if isinstance(action, np.ndarray):
                    action = tuple(action.astype(int))
                else:
                    action = tuple(action)
                self.actionReady.emit(action)
            except Exception as e:
                print(f"RLWorker error: {e}")
            QThread.msleep(50)

# -----------------------------
# Board
# -----------------------------
class Board(QFrame):
    msg2Statusbar = pyqtSignal(str)

    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.gridSize = gridSize
        self.setFixedSize(gridSize * BOARD_DATA.width, gridSize * BOARD_DATA.height)
        self.score = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw all board cells
        for x in range(BOARD_DATA.width):
            for y in range(BOARD_DATA.height):
                val = BOARD_DATA.getValue(x, y)
                drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)
        # Draw current shape
        for x, y in BOARD_DATA.getCurrentShapeCoord():
            val = BOARD_DATA.currentShape.shape
            drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw game over text
        if hasattr(self.window(), "isGameOver") and self.window().isGameOver:
            painter.setPen(QColor(255, 0, 0))
            painter.setFont(QFont("Arial", 20, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "Game Over!")

    def updateData(self):
        self.msg2Statusbar.emit(str(self.score))
        self.update()

# -----------------------------
# Side Panel
# -----------------------------
class SidePanel(QFrame):
    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.gridSize = gridSize
        self.setFixedSize(gridSize * 5, gridSize * BOARD_DATA.height)
        self.move(gridSize * BOARD_DATA.width, 0)

    def updateData(self):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0x000000))
        next_shape = BOARD_DATA.nextShape
        if next_shape.shape != Shape.shapeNone:
            offset_x = 2
            offset_y = BOARD_DATA.height // 2 - 2
            for x, y in next_shape.getCoords(0, 0, 0):
                drawSquare(painter, (x + offset_x) * self.gridSize,
                           (y + offset_y) * self.gridSize,
                           next_shape.shape, self.gridSize)
        painter.setPen(QColor(0x777777))
        painter.drawLine(self.width()-1, 0, self.width()-1, self.height())
        painter.setPen(QColor(0xCCCCCC))
        painter.drawLine(self.width(), 0, self.width(), self.height())

# -----------------------------
# Draw square
# -----------------------------
def drawSquare(painter, x, y, val, s):
    colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                  0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]
    color = QColor(colorTable[val]) if val < len(colorTable) else QColor(0xFFFFFF)
    painter.fillRect(x + 1, y + 1, s - 2, s - 2, color)
    painter.setPen(QColor(0x000000))
    painter.drawRect(x, y, s, s)

# -----------------------------
# Game chính Tetris
# -----------------------------
class Tetris(QMainWindow):
    def __init__(self, mode="Heuristic"):
        super().__init__()
        self.isStarted = False
        self.isPaused = False
        self.isGameOver = False
        self.nextMove = None
        self.rl_action = None
        self.mode = mode
        self.last_shape = None

        self.rl_agent = None
        self.rl_thread = None
        self.rl_worker = None
        self.env = None

        # Lưu bước chơi Heuristic
        self.heuristic_steps = []

        # Quản lý episode Heuristic
        self.episode = 0
        self.max_episodes = 50
        self.episode_start_time = None
        self.episode_timeout = 60  # 1 phút

        # Dataset file
        self.dataset_file = 'heuristic_dataset.npy'
        self.log_file = 'heuristic.txt'
        if not os.path.exists(self.dataset_file):
            np.save(self.dataset_file, np.array([], dtype=object))

        # Khởi tạo env cho tất cả các mode
        self.env = TetrisEnv()

        if self.mode in ["DQN", "PPO"]:
            self.init_rl_agent()

        self.initUI()

    # -----------------------------
    # UI
    # -----------------------------
    def initUI(self):
        self.gridSize = 22
        self.speed = 200
        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)

        mainLayout = QVBoxLayout()
        hLayout = QHBoxLayout()

        self.tboard = Board(self, self.gridSize)
        hLayout.addWidget(self.tboard)

        self.sidePanel = SidePanel(self, self.gridSize)
        hLayout.addWidget(self.sidePanel)

        mainLayout.addLayout(hLayout)

        buttonLayout = QHBoxLayout()
        self.startButton = QPushButton("Start")
        self.pauseButton = QPushButton("Pause")
        self.modeButton = QPushButton(f"{self.mode} Mode")

        self.startButton.clicked.connect(self.start)
        self.pauseButton.clicked.connect(self.pause)
        self.modeButton.clicked.connect(self.toggle_mode)

        buttonLayout.addWidget(self.startButton)
        buttonLayout.addWidget(self.pauseButton)
        buttonLayout.addWidget(self.modeButton)

        mainLayout.addLayout(buttonLayout)

        centralWidget = QFrame()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)
        self.statusbar = self.statusBar()
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.center()
        self.setWindowTitle("Tetris Multi-AI Mode")
        self.show()
        self.setFixedSize(self.tboard.width() + self.sidePanel.width() + 50,
                          self.sidePanel.height() + self.statusbar.height() + 100)

    # -----------------------------
    # Khởi tạo agent RL
    # -----------------------------
    def init_rl_agent(self):
        if self.mode == "DQN":
            self.rl_agent = DQNAgent(state_shape=(1, BOARD_DATA.height, BOARD_DATA.width))
            model_path = "models/final_model.pth"
            if os.path.exists(model_path):
                self.rl_agent.load(model_path)
        elif self.mode == "PPO":
            self.rl_agent = PPOAgent(state_shape=(1, BOARD_DATA.height, BOARD_DATA.width))
            model_path = "models/final_ppo_model.pth"
            if os.path.exists(model_path):
                self.rl_agent.load(model_path)
        self.start_rl_thread()

    def start_rl_thread(self):
        if self.rl_agent:
            self.rl_worker = RLWorker(self.rl_agent, self.env, self.mode)
            self.rl_thread = QThread()
            self.rl_worker.moveToThread(self.rl_thread)
            self.rl_thread.started.connect(self.rl_worker.run)
            self.rl_worker.actionReady.connect(self.set_rl_action)
            self.rl_thread.start()

    def stop_rl_thread(self):
        if self.rl_worker:
            self.rl_worker.running = False
        if self.rl_thread:
            self.rl_thread.quit()
            self.rl_thread.wait()
        self.rl_thread = None
        self.rl_worker = None

    def set_rl_action(self, action):
        self.rl_action = action

    # -----------------------------
    # Chuyển mode
    # -----------------------------
    def toggle_mode(self):
        modes = ["Heuristic", "DQN", "PPO"]
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        self.modeButton.setText(f"{self.mode} Mode")
        self.statusbar.showMessage(f"Switched to {self.mode} mode")
        self.stop_rl_thread()
        self.nextMove = None
        self.last_shape = None
        if self.mode in ["DQN", "PPO"]:
            self.init_rl_agent()
        elif self.mode == "Heuristic":
            self.env = TetrisEnv()  # luôn đảm bảo env có sẵn

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)

    # -----------------------------
    # Start / Pause
    # -----------------------------
    def start(self):
        if self.mode == "Heuristic":
            self.start_episode()
        else:
            self.isPaused = False
            self.isStarted = True
            self.isGameOver = False
            self.tboard.score = 0
            BOARD_DATA.clear()
            BOARD_DATA.createNewPiece()
            self.timer.start(self.speed, self)
            self.nextMove = None
            self.last_shape = None
            self.heuristic_steps = []

    def start_episode(self):
        self.episode += 1
        if self.episode > self.max_episodes:
            self.timer.stop()
            print("✅ Finished all episodes")
            return

        self.isPaused = False
        self.isGameOver = False
        self.tboard.score = 0
        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()
        self.timer.start(self.speed, self)
        self.nextMove = None
        self.last_shape = None
        self.heuristic_steps = []
        self.episode_start_time = time.time()
        print(f"🚀 Starting episode {self.episode}")

    def pause(self):
        if not self.isStarted or self.isGameOver:
            return
        self.isPaused = not self.isPaused
        if self.isPaused:
            self.timer.stop()
            self.tboard.msg2Statusbar.emit("Paused")
        else:
            self.timer.start(self.speed, self)
        self.updateWindow()

    def updateWindow(self):
        self.tboard.updateData()
        self.sidePanel.updateData()
        self.update()

    # -----------------------------
    # Timer event
    # -----------------------------
    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            super(Tetris, self).timerEvent(event)
            return
        if self.isGameOver:
            self.timer.stop()
            return

        # Timeout Heuristic episode
        if self.mode == "Heuristic" and self.episode_start_time:
            if time.time() - self.episode_start_time > self.episode_timeout:
                print("⏱ Episode timeout, restarting...")
                self.save_heuristic_dataset()
                self.start_episode()
                return

        if self.mode in ["DQN", "PPO"] and self.rl_agent:
            if self.rl_action is not None:
                self.env.step(self.rl_action)
                self.rl_action = None
                self.tboard.score = BOARD_DATA.removed_lines * 10
        else:
            # Heuristic mode
            if TETRIS_AI:
                if self.last_shape != BOARD_DATA.currentShape.shape:
                    self.nextMove = TETRIS_AI.nextMove()
                    self.last_shape = BOARD_DATA.currentShape.shape

            if self.nextMove:
                k = 0
                while BOARD_DATA.currentDirection != self.nextMove[0] and k < 4:
                    BOARD_DATA.rotateRight()
                    k += 1
                k = 0
                while BOARD_DATA.currentX != self.nextMove[1] and k < BOARD_DATA.width:
                    if BOARD_DATA.currentX > self.nextMove[1]:
                        BOARD_DATA.moveLeft()
                    elif BOARD_DATA.currentX < self.nextMove[1]:
                        BOARD_DATA.moveRight()
                    k += 1

            reward = BOARD_DATA.moveDown()
            self.tboard.score += reward

            # Lưu step heuristic dạng dictionary
            if self.env:
                step = {
                    'state': self.env._get_observation(),
                    'action': self.nextMove,
                    'reward': reward,
                    'next_state': self.env._get_observation(),
                    'done': BOARD_DATA.gameOver()
                }
                self.heuristic_steps.append(step)

        if BOARD_DATA.gameOver():
            self.gameOver()
            return

        self.updateWindow()

    # -----------------------------
    # Key press
    # -----------------------------
    def keyPressEvent(self, event):
        if not self.isStarted or BOARD_DATA.currentShape == Shape.shapeNone:
            super(Tetris, self).keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_P:
            self.pause()
        elif self.isPaused:
            return
        elif key == Qt.Key_Left:
            BOARD_DATA.moveLeft()
        elif key == Qt.Key_Right:
            BOARD_DATA.moveRight()
        elif key == Qt.Key_Up:
            BOARD_DATA.rotateLeft()
        elif key == Qt.Key_Space:
            self.tboard.score += BOARD_DATA.dropDown()
        elif key == Qt.Key_R:
            self.start_episode()
        elif key == Qt.Key_M:
            self.toggle_mode()
        else:
            super(Tetris, self).keyPressEvent(event)
        self.updateWindow()

    # -----------------------------
    # Game over
    # -----------------------------
    def gameOver(self):
        self.isGameOver = True
        self.tboard.msg2Statusbar.emit("Game Over!")
        self.updateWindow()
        if self.mode == "Heuristic":
            self.save_heuristic_dataset()
            self.start_episode()
        else:
            self.stop_rl_thread()

    # -----------------------------
    # Lưu dataset heuristic
    # -----------------------------
    def save_heuristic_dataset(self):
        if not self.heuristic_steps:
            return
        old_data = np.load(self.dataset_file, allow_pickle=True)
        old_data = old_data.tolist() if old_data.size else []
        old_data.extend(self.heuristic_steps)
        np.save(self.dataset_file, np.array(old_data, dtype=object))

        # Lưu log
        with open(self.log_file, 'a') as f:
            f.write(f'Episode {self.episode}: steps={len(self.heuristic_steps)}, score={self.tboard.score}\n')
        print(f"💾 Episode {self.episode} saved: {len(self.heuristic_steps)} steps, score={self.tboard.score}")
        self.heuristic_steps = []

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    app = QApplication([])
    tetris = Tetris(mode="Heuristic")
    sys.exit(app.exec_())
