"""
test suite for the tetris implementation
"""
import pytest
import numpy as np
from tetris import TetrisEnv, Action, Tetromino, TetrominoType, Board


def test_tetromino_creation():
    """test basic tetromino creation and properties"""
    piece = Tetromino(TetrominoType.I)
    assert piece.type == TetrominoType.I
    assert piece.x == 0
    assert piece.y == 0
    assert piece.rotation == 0
    assert piece.shape.shape == (1, 4)  # I piece is 1x4


def test_tetromino_rotation():
    """test tetromino rotation mechanics"""
    piece = Tetromino(TetrominoType.T)
    original_shape = piece.shape.copy()
    
    # test clockwise rotation
    piece.rotate(clockwise=True)
    assert not np.array_equal(piece.shape, original_shape)
    assert piece.rotation == 1
    
    # test that four rotations return to original shape
    for _ in range(3):
        piece.rotate(clockwise=True)
    assert np.array_equal(piece.shape, original_shape)
    assert piece.rotation == 0


def test_board_initialization():
    """test game board initialization"""
    board = Board(width=10, height=20)
    assert board.width == 10
    assert board.height == 20
    assert board.grid.shape == (20, 10)
    assert not board.game_over
    assert board.score == 0
    assert board.lines_cleared == 0
    assert board.current_piece is not None
    assert board.next_piece is not None


def test_board_piece_movement():
    """test piece movement on the board"""
    board = Board(width=10, height=20)
    
    # test left movement
    initial_x = board.current_piece.x
    board.move_left()
    assert board.current_piece.x == initial_x - 1
    
    # test right movement
    board.move_right()
    assert board.current_piece.x == initial_x
    
    # test down movement
    initial_y = board.current_piece.y
    board.move_down()
    assert board.current_piece.y == initial_y + 1


def test_environment_initialization():
    """test environment initialization"""
    env = TetrisEnv()
    obs, info = env.reset()
    
    assert obs.shape == (20, 10)  # default board size
    assert isinstance(info, dict)
    assert 'score' in info
    assert 'lines_cleared' in info
    assert 'game_over' in info
    assert 'next_piece' in info


def test_environment_step():
    """test environment step function"""
    env = TetrisEnv()
    obs, info = env.reset()
    
    # test each action
    for action in Action:
        obs, reward, terminated, truncated, info = env.step(action.value)
        assert obs.shape == (20, 10)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


def test_line_clearing():
    """test line clearing mechanics"""
    board = Board(width=4, height=4)
    
    # create a full line at the bottom
    board.grid[3] = np.array([ord('I')] * 4)
    
    # place a piece that will clear the line
    board.current_piece = Tetromino(TetrominoType.I)
    board.current_piece.x = 0
    board.current_piece.y = 2
    
    # land the piece
    board._land_piece()
    
    assert board.lines_cleared == 1
    assert board.score > 0


def test_game_over_condition():
    """test game over condition"""
    board = Board(width=4, height=4)
    
    # fill the board to force game over
    board.grid.fill(ord('I'))
    
    # try to spawn a new piece
    assert not board._spawn_current_piece()
    assert board.game_over


if __name__ == '__main__':
    pytest.main([__file__]) 