from tetris_model import BOARD_DATA, Shape
import numpy as np
import math
from datetime import datetime


class TetrisAI(object):
    def nextMove(self):
        """Chọn (rotation, x, score) tốt nhất cho current shape (có lookahead next shape)"""
        if BOARD_DATA.currentShape == Shape.shapeNone:
            return None

        strategy = None

        # Rotation hợp lệ cho current shape
        if BOARD_DATA.currentShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            d0Range = (0, 1)
        elif BOARD_DATA.currentShape.shape == Shape.shapeO:
            d0Range = (0,)
        else:
            d0Range = (0, 1, 2, 3)

        # Rotation hợp lệ cho next shape
        if BOARD_DATA.nextShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            d1Range = (0, 1)
        elif BOARD_DATA.nextShape.shape == Shape.shapeO:
            d1Range = (0,)
        else:
            d1Range = (0, 1, 2, 3)

        # Thử hết current + next shape
        for d0 in d0Range:
            minX, maxX, _, _ = BOARD_DATA.currentShape.getBoundingOffsets(d0)
            for x0 in range(-minX, BOARD_DATA.width - maxX):
                board = self.calcStep1Board(d0, x0)

                for d1 in d1Range:
                    minX, maxX, _, _ = BOARD_DATA.nextShape.getBoundingOffsets(d1)
                    dropDist = self.calcNextDropDist(board, d1, range(-minX, BOARD_DATA.width - maxX))

                    for x1 in range(-minX, BOARD_DATA.width - maxX):
                        score = self.calculateScore(np.copy(board), d1, x1, dropDist)
                        if not strategy or strategy[2] < score:
                            strategy = (d0, x0, score)

        return strategy

    # ------------------------------------------------------

    def calcNextDropDist(self, data, d0, xRange):
        """Khoảng rơi của next shape"""
        res = {}
        for x0 in xRange:
            res[x0] = BOARD_DATA.height - 1
            for x, y in BOARD_DATA.nextShape.getCoords(d0, x0, 0):
                yy = 0
                while yy + y < BOARD_DATA.height and (
                    yy + y < 0 or data[(y + yy), x] == Shape.shapeNone
                ):
                    yy += 1
                yy -= 1
                if yy < res[x0]:
                    res[x0] = yy
        return res

    def calcStep1Board(self, d0, x0):
        """Board sau khi đặt current shape"""
        board = np.array(BOARD_DATA.getData()).reshape(
            (BOARD_DATA.height, BOARD_DATA.width)
        )
        self.dropDown(board, BOARD_DATA.currentShape, d0, x0)
        return board

    def dropDown(self, data, shape, direction, x0):
        """Thả current shape"""
        dy = BOARD_DATA.height - 1
        for x, y in shape.getCoords(direction, x0, 0):
            yy = 0
            while yy + y < BOARD_DATA.height and (
                yy + y < 0 or data[(y + yy), x] == Shape.shapeNone
            ):
                yy += 1
            yy -= 1
            if yy < dy:
                dy = yy
        self.dropDownByDist(data, shape, direction, x0, dy)

    def dropDownByDist(self, data, shape, direction, x0, dist):
        """Thả shape xuống vị trí cụ thể"""
        for x, y in shape.getCoords(direction, x0, 0):
            if 0 <= y + dist < BOARD_DATA.height:  # safeguard tránh crash
                data[y + dist, x] = shape.shape

    # ------------------------------------------------------

    def calculateScore(self, step1Board, d1, x1, dropDist):
        """Tính điểm heuristic (y hệt bản cũ bạn gửi)"""
        width, height = BOARD_DATA.width, BOARD_DATA.height

        # Thả next shape
        self.dropDownByDist(step1Board, BOARD_DATA.nextShape, d1, x1, dropDist[x1])

        fullLines = 0
        roofY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width
        vBlocks = 0

        # Duyệt từ dưới lên
        for y in range(height - 1, -1, -1):
            hasHole, hasBlock = False, False
            for x in range(width):
                if step1Board[y, x] == Shape.shapeNone:
                    hasHole = True
                    holeCandidates[x] += 1
                else:
                    hasBlock = True
                    roofY[x] = height - y
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]
                        holeCandidates[x] = 0
                    if holeConfirm[x] > 0:
                        vBlocks += 1
            if not hasBlock:
                break
            if not hasHole and hasBlock:
                fullLines += 1

        vHoles = sum([x ** 0.7 for x in holeConfirm])
        maxHeight = max(roofY) - fullLines

        roofDy = [roofY[i] - roofY[i + 1] for i in range(len(roofY) - 1)]

        stdDY = (
            math.sqrt(
                sum([dy ** 2 for dy in roofDy]) / len(roofDy)
                - (sum(roofDy) / len(roofDy)) ** 2
            )
            if roofDy
            else 0
        )

        absDy = sum(abs(d) for d in roofDy)
        maxDy = max(roofY) - min(roofY) if roofY else 0

        # Công thức scoring gốc
        score = (
            fullLines * 1.8
            - vHoles * 1.0
            - vBlocks * 0.5
            - maxHeight ** 1.5 * 0.02
            - stdDY * 0.01
            - absDy * 0.2
            - maxDy * 0.3
        )
        return score


# Singleton AI
TETRIS_AI = TetrisAI()
