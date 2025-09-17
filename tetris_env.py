import numpy as np
from tetris_model import BOARD_DATA, Shape

class TetrisEnv:
    """
    Environment cho RL agent & Heuristic.
    Observation: (1, H, W) float32
    Action: tuple (rotation 0-3, x position 0..width-1)
    Reward: lines cleared, holes, bumpiness, height
    """
    def __init__(self):
        self.board = BOARD_DATA
        self.width = BOARD_DATA.width
        self.height = BOARD_DATA.height
        self.rotation_size = 4

        self.current_shape = self.board.currentShape.shape
        self.next_shape = self.board.nextShape.shape

    def reset(self):
        """Reset lại game về trạng thái ban đầu"""
        self.board.clear()
        self.board.gameOverFlag = False  # đảm bảo chưa game over
        self.board.createNewPiece()

        self.current_shape = self.board.currentShape.shape
        self.next_shape = self.board.nextShape.shape

        return self._get_observation(), {
            "current_shape": self.current_shape,
            "next_shape": self.next_shape
        }

    def _get_observation(self):
        obs = np.zeros((self.height, self.width), dtype=np.float32)
        for x in range(self.width):
            for y in range(self.height):
                obs[y, x] = self.board.getValue(x, y)
        return obs[np.newaxis, :, :]

    def step(self, action):
        """Thực hiện 1 bước action: (rotation, x position)"""
        try:
            rotation_target, x_target = action[0], action[1]

            # Xoay về hướng mong muốn
            if self.board.currentDirection != rotation_target:
                self.board.rotateRight()

            # Di chuyển sang trái/phải
            if self.board.currentX < x_target:
                self.board.moveRight()
            elif self.board.currentX > x_target:
                self.board.moveLeft()

            # Thả xuống
            lines_cleared = 0
            if not self.board.gameOver():
                lines_cleared = self.board.moveDown()

            # Reward
            reward = lines_cleared * 10
            reward -= self.board.get_holes() * 5
            reward -= self.board.get_aggregate_height() * 0.2
            reward -= self.board.get_bumpiness() * 0.1
            if self.board.gameOver():
                reward -= 1000

            terminated = self.board.gameOver()
            truncated = False

            self.current_shape = self.board.currentShape.shape
            self.next_shape = self.board.nextShape.shape

            info = {
                "current_shape": self.current_shape,
                "next_shape": self.next_shape
            }

            return self._get_observation(), reward, terminated, truncated, info

        except Exception as e:
            print(f"TetrisEnv.step() error: {e}")
            return self._get_observation(), -100, True, False, {}

    def render(self):
        """In board ra console"""
        board = self._get_observation()[0]
        print("-" * self.width)
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if board[y, x] == Shape.shapeNone:
                    row += "."
                else:
                    row += str(int(board[y, x]))
            print(row)
        print("-" * self.width)
