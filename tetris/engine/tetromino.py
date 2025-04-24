"""
tetromino stuff for the tetris game engine
"""
from enum import Enum
import random
import numpy as np


class TetrominoType(Enum):
    """all the different tetromino shapes we can have"""
    I = 'I'
    O = 'O'
    T = 'T'
    S = 'S'
    Z = 'Z'
    J = 'J'
    L = 'L'


class Tetromino:
    """
    handles a tetromino piece w its shape and how it rotates
    """
    
    # shapes for each type using numpy arrays
    SHAPES: dict[TetrominoType, np.ndarray] = {
        TetrominoType.I: np.array([[1, 1, 1, 1]], dtype=np.int8),
        TetrominoType.O: np.array([[1, 1],
                                  [1, 1]], dtype=np.int8),
        TetrominoType.T: np.array([[0, 1, 0],
                                  [1, 1, 1]], dtype=np.int8),
        TetrominoType.S: np.array([[0, 1, 1],
                                  [1, 1, 0]], dtype=np.int8),
        TetrominoType.Z: np.array([[1, 1, 0],
                                  [0, 1, 1]], dtype=np.int8),
        TetrominoType.J: np.array([[1, 0, 0],
                                  [1, 1, 1]], dtype=np.int8),
        TetrominoType.L: np.array([[0, 0, 1],
                                  [1, 1, 1]], dtype=np.int8)
    }

    # colors for each type (rgb)
    COLORS: dict[TetrominoType, tuple[int, int, int]] = {
        TetrominoType.I: (0, 240, 240),    # cyan
        TetrominoType.O: (240, 240, 0),    # yellow
        TetrominoType.T: (160, 0, 240),    # purple
        TetrominoType.S: (0, 240, 0),      # green
        TetrominoType.Z: (240, 0, 0),      # red
        TetrominoType.J: (0, 0, 240),      # blue
        TetrominoType.L: (240, 160, 0),    # orange
    }

    def __init__(self, piece_type: TetrominoType):
        """
        make a new tetromino piece
        
        args:
            piece_type: what kinda piece to make
        """
        self.type = piece_type
        self.shape = self.SHAPES[piece_type].copy()
        self.rotation = 0  # 0=0deg, 1=90deg, 2=180deg, 3=270deg
        self.x = 0  # where piece is on board (x)
        self.y = 0  # where piece is on board (y)
        self.color = self.COLORS[piece_type]

    def rotate(self, clockwise: bool = True) -> None:
        """
        spin the piece 90 degrees
        
        args:
            clockwise: true for clockwise, false for counterclockwise
        """
        self.shape = np.rot90(self.shape, k=(3 if clockwise else 1))
        self.rotation = (self.rotation + (1 if clockwise else -1)) % 4

    def get_positions(self) -> list[tuple[int, int]]:
        """
        get where all blocks in the piece are relative to (x,y)
        
        returns:
            list of (x,y) spots for each block in piece
        """
        positions: list[tuple[int, int]] = []
        for row in range(self.shape.shape[0]):
            for col in range(self.shape.shape[1]):
                if self.shape[row][col]:
                    positions.append((self.x + col, self.y + row))
        return positions

    @classmethod
    def random(cls) -> 'Tetromino':
        """
        make a random piece
        
        returns:
            new random tetromino
        """
        piece_type = random.choice(list(TetrominoType))
        return cls(piece_type) 