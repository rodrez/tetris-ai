"""
game board implementation for tetris game engine
"""
import numpy as np
from .tetromino import Tetromino


class Board:
    """
    handles the tetris game board and keeps track of whats happening
    """

    def __init__(self, width: int = 10, height: int = 20):
        """
        make a new game board
        
        args:
            width: how wide the board is (default: 10)
            height: how tall the board is (default: 20)
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.current_piece: Tetromino | None = None
        self.next_piece: Tetromino | None = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        # setup first pieces
        self._spawn_next_piece()
        self._spawn_current_piece()

    def copy(self) -> 'Board':
        """make a deep copy of board state"""
        new_board = Board(self.width, self.height)
        new_board.grid = self.grid.copy()
        new_board.score = self.score
        new_board.lines_cleared = self.lines_cleared
        new_board.game_over = self.game_over
        
        # copy pieces if they exist
        if self.current_piece:
            new_board.current_piece = Tetromino(self.current_piece.type)
            new_board.current_piece.shape = self.current_piece.shape.copy()
            new_board.current_piece.x = self.current_piece.x
            new_board.current_piece.y = self.current_piece.y
            new_board.current_piece.rotation = self.current_piece.rotation
            
        if self.next_piece:
            new_board.next_piece = Tetromino(self.next_piece.type)
            new_board.next_piece.shape = self.next_piece.shape.copy()
            new_board.next_piece.x = self.next_piece.x
            new_board.next_piece.y = self.next_piece.y
            new_board.next_piece.rotation = self.next_piece.rotation
            
        return new_board

    def _spawn_next_piece(self) -> None:
        """make the next piece that'll drop"""
        self.next_piece = Tetromino.random()

    def _spawn_current_piece(self) -> bool:
        """
        move next piece to current and make a new next piece
        
        returns:
            bool: false if piece cant be placed (game over), true if all good
        """
        if self.next_piece is None:
            self._spawn_next_piece()
        
        self.current_piece = self.next_piece
        self._spawn_next_piece()
        
        # put piece at top center of board
        if self.current_piece:
            self.current_piece.x = (self.width - self.current_piece.shape.shape[1]) // 2
            self.current_piece.y = 0
            
            # check if piece can fit
            if not self._is_valid_position():
                self.game_over = True
                return False
        return True

    def _is_valid_position(self) -> bool:
        """
        check if current piece can be where it is
        
        returns:
            bool: true if position works, false if not
        """
        if not self.current_piece:
            return True
            
        positions = self.current_piece.get_positions()
        for x, y in positions:
            # check if in bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                return False
            # check if hits other pieces
            if y >= 0 and self.grid[y][x] != 0:
                return False
        return True

    def move_left(self) -> bool:
        """
        try to move piece left
        
        returns:
            bool: true if piece moved, false if cant
        """
        if not self.current_piece or self.game_over:
            return False
            
        self.current_piece.x -= 1
        if not self._is_valid_position():
            self.current_piece.x += 1
            return False
        return True

    def move_right(self) -> bool:
        """
        try to move piece right
        
        returns:
            bool: true if piece moved, false if cant
        """
        if not self.current_piece or self.game_over:
            return False
            
        self.current_piece.x += 1
        if not self._is_valid_position():
            self.current_piece.x -= 1
            return False
        return True

    def rotate(self, clockwise: bool = True) -> bool:
        """
        try to rotate the piece
        
        args:
            clockwise: true for clockwise, false for counterclockwise
            
        returns:
            bool: true if piece rotated, false if cant
        """
        if not self.current_piece or self.game_over:
            return False
            
        original_shape = self.current_piece.shape.copy()
        self.current_piece.rotate(clockwise)
        
        if not self._is_valid_position():
            self.current_piece.shape = original_shape
            return False
        return True

    def move_down(self) -> bool:
        """
        try to move piece down
        
        returns:
            bool: true if piece moved, false if it landed
        """
        if not self.current_piece or self.game_over:
            return False
            
        self.current_piece.y += 1
        if not self._is_valid_position():
            self.current_piece.y -= 1
            self._land_piece()
            return False
        return True

    def hard_drop(self) -> None:
        """drop piece straight to bottom"""
        if not self.current_piece or self.game_over:
            return
            
        while self.move_down():
            pass

    def _land_piece(self) -> None:
        """
        stick the piece to board and clear any full lines
        """
        if not self.current_piece:
            return
            
        # add piece to grid
        positions = self.current_piece.get_positions()
        piece_type_value = self.current_piece.type.value
        for x, y in positions:
            if 0 <= y < self.height:  # only place pieces in bounds
                self.grid[y][x] = ord(piece_type_value)  # use ascii value of piece type
                
        # find completed lines
        lines_to_clear = []
        for y in range(self.height):
            if np.all(self.grid[y] != 0):
                lines_to_clear.append(y)
                
        if lines_to_clear:
            self._clear_lines(lines_to_clear)
            
        # get next piece ready
        self._spawn_current_piece()

    def _clear_lines(self, lines: list[int]) -> None:
        """
        clear lines and update score
        
        args:
            lines: which lines to clear (y coords)
        """
        # remove lines and add empty ones at top
        self.grid = np.delete(self.grid, lines, axis=0)
        self.grid = np.vstack([np.zeros((len(lines), self.width), dtype=np.int8), self.grid])
        
        # update score n lines cleared
        self.lines_cleared += len(lines)
        # more points for clearing multiple lines at once
        points = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += points.get(len(lines), 0)

    def get_state(self) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.int8]], Tetromino | None, Tetromino | None, int, int, bool]:
        """
        get current game state
        
        returns:
            tuple w:
            - game grid
            - current piece
            - next piece
            - score
            - lines cleared
            - if game over
        """
        return (
            self.grid.copy(),
            self.current_piece,
            self.next_piece,
            self.score,
            self.lines_cleared,
            self.game_over
        ) 