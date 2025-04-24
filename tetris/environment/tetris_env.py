"""
tetris env that follows the openai gym pattern
"""
from enum import IntEnum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, cast
from numpy.typing import NDArray
from ..engine.board import Board


class Action(IntEnum):
    """stuff u can do in the tetris env"""
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6


class TetrisEnv(gym.Env[NDArray[np.uint8], np.int64]):
    """
    tetris env using gymnasium interface
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    # how we score different stuff
    REWARD_WEIGHTS = {
        # points for clearing lines
        'lines_cleared': 100.0,      # base points for each line
        'tetris_bonus': 300.0,       # extra points for clearing 4 at once
        'combo_bonus': 50.0,         # bonus for clearing lines in a row
        
        # penalties for messy board
        'height_penalty': -1.0,      # lose points for going too high
        'hole_penalty': -10.0,       # lose points for making holes
        'bumpiness_penalty': -2.0,   # lose points for uneven surface
        'game_over': -500.0,         # big penalty for losing
        
        # points for being efficient
        'move_penalty': -0.1,        # tiny penalty for each move
        'landing_height_penalty': -1.0,  # lose points for placing pieces high up
        
        # points for good structure
        'well_bonus': 10.0,          # points for keeping a well for tetris
        'flat_bonus': 5.0,           # points for flat surface bits
        'clear_path_bonus': 2.0,     # points for paths to holes
        'proper_well_placement': 20.0,  # points for keeping well structure nice
        
        # fancy move bonuses
        'tspin_setup_bonus': 30.0,   # points for setting up t-spins
        'perfect_clear_bonus': 500.0, # big points for clearing whole board
        'overhang_penalty': -5.0,    # lose points for dangly bits
        'blocked_well_penalty': -15.0,  # lose points for messing up the well
    }

    def __init__(self, width: int = 10, height: int = 20, render_mode: str | None = None):
        """
        start up the tetris env
        
        args:
            width: how wide the board is (default: 10)
            height: how tall the board is (default: 20)
            render_mode: how to show the game (default: None)
        """
        super().__init__()
        
        self.width: int = width
        self.height: int = height
        self.render_mode: str | None = render_mode
        self.board: Board = Board(width, height)
        self.combo_count: int = 0  # keep track of line clear combos
        
        # setup what actions we can do
        self.action_space = spaces.Discrete(len(Action))
        
        # setup what the board looks like
        # board is 2d grid where each cell is:
        # 0 for empty, or ascii values for piece types
        self.observation_space = spaces.Box(
            low=0,
            high=255,  # max ascii value
            shape=(height, width),
            dtype=np.uint8
        )
        
        # setup renderer if needed
        self.renderer = None
        if render_mode == "human":
            from ..visualization.renderer import TetrisRenderer
            self.renderer = TetrisRenderer(width, height)

    def _get_holes(self, grid: NDArray[np.int8 | np.uint8]) -> int:
        """
        count holes in the board
        a hole is an empty spot w filled spots above it
        
        args:
            grid: the game board
            
        returns:
            how many holes we found
        """
        grid = grid.astype(np.uint8)
        holes = 0
        # check each col
        for x in range(self.width):
            found_block = False
            # look from top to bottom
            for y in range(self.height):
                if grid[y][x] != 0:  # found a block
                    found_block = True
                elif found_block and grid[y][x] == 0:  # found a hole
                    holes += 1
        return holes

    def _get_bumpiness_and_height(self, grid: NDArray[np.int8 | np.uint8]) -> tuple[float, list[int]]:
        """
        check how bumpy the board is and how high stuff is
        bumpiness is how much heights change between cols
        
        args:
            grid: the game board
            
        returns:
            tuple w:
            - how bumpy it is
            - list of col heights
        """
        grid = grid.astype(np.uint8)
        heights = []
        # get height of each col
        for x in range(self.width):
            for y in range(self.height):
                if grid[y][x] != 0:
                    heights.append(self.height - y)
                    break
            else:
                heights.append(0)
        
        # calc how bumpy it is
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
            
        return bumpiness, heights

    def _get_landing_height(self, grid: NDArray[np.int8 | np.uint8], 
                          prev_grid: NDArray[np.int8 | np.uint8]) -> int:
        """
        see how high the last piece landed
        
        args:
            grid: current board state
            prev_grid: board state before piece landed
            
        returns:
            how high the piece landed (0 if we cant tell)
        """
        grid = grid.astype(np.uint8)
        prev_grid = prev_grid.astype(np.uint8)
        diff = grid - prev_grid
        if np.any(diff):
            # find highest changed spot
            for y in range(self.height):
                if np.any(diff[y] != 0):
                    return self.height - y
        return 0

    def _analyze_well_structure(self, grid: NDArray[np.uint8]) -> tuple[bool, int]:
        """
        check if we got a good well for tetris
        well is a col w higher cols next to it, good for tetris
        
        args:
            grid: the game board
            
        returns:
            tuple w:
            - if we got a good well
            - how good the well is (depth n cleanness)
        """
        well_exists = False
        well_quality = 0
        
        # check edges and one in from edges
        potential_well_columns = [0, 1, self.width-2, self.width-1]
        
        for x in potential_well_columns:
            empty_count = 0
            adjacent_filled = True
            
            # see if cols next to it r higher
            for y in range(self.height-1, -1, -1):
                if x > 0 and grid[y][x-1] == 0:  # left side empty
                    adjacent_filled = False
                if x < self.width-1 and grid[y][x+1] == 0:  # right side empty
                    adjacent_filled = False
                
                if grid[y][x] == 0:
                    empty_count += 1
                else:
                    break
            
            if empty_count >= 4 and adjacent_filled:
                well_exists = True
                well_quality = empty_count
                break
                
        return well_exists, well_quality

    def _detect_flat_surfaces(self, grid: NDArray[np.uint8]) -> list[tuple[int, int]]:
        """
        find flat bits on surface
        
        args:
            grid: the game board
            
        returns:
            list of (where it starts, how long it is) for flat bits
        """
        flat_segments = []
        heights = []
        
        # get height of each col
        for x in range(self.width):
            for y in range(self.height):
                if grid[y][x] != 0:
                    heights.append(self.height - y)
                    break
            else:
                heights.append(0)
        
        # look for flat bits
        start_x = 0
        current_height = heights[0]
        
        for x in range(1, self.width):
            if heights[x] != current_height:
                if x - start_x >= 2:  # need at least 2 blocks to be flat
                    flat_segments.append((start_x, x - start_x))
                start_x = x
                current_height = heights[x]
        
        # check last bit
        if self.width - start_x >= 2 and all(h == current_height for h in heights[start_x:]):
            flat_segments.append((start_x, self.width - start_x))
            
        return flat_segments

    def _evaluate_tspin_potential(self, grid: NDArray[np.uint8]) -> float:
        """
        check how good spots r for t-spins
        
        args:
            grid: the game board
            
        returns:
            score for t-spin friendly spots
        """
        tspin_score = 0.0
        
        # look for t-spin patterns
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                # check common t-spin spots
                if grid[y][x] == 0:  # empty middle
                    corners_filled = sum(1 for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]
                                      if 0 <= x+dx < self.width and 0 <= y+dy < self.height 
                                      and grid[y+dy][x+dx] != 0)
                    if corners_filled >= 3:  # might be good for t-spin
                        tspin_score += 1.0
                        
                        # extra points if we can get a t piece in
                        if (y > 0 and all(grid[y-1][x+dx] == 0 for dx in [-1,0,1])):
                            tspin_score += 0.5
                            
        return tspin_score

    def _check_overhangs(self, grid: NDArray[np.uint8]) -> int:
        """
        count dangly bits in board
        dangly bit is empty spot w filled spot above
        
        args:
            grid: the game board
            
        returns:
            how many dangly bits we found
        """
        overhangs = 0
        
        for x in range(self.width):
            for y in range(1, self.height):
                if grid[y][x] == 0 and grid[y-1][x] != 0:
                    overhangs += 1
                    
        return overhangs

    def _analyze_piece_placement(self, grid: NDArray[np.uint8], 
                               prev_grid: NDArray[np.uint8]) -> dict[str, float]:
        """
        check how good the last piece placement was
        
        args:
            grid: current board state
            prev_grid: board state before piece
            
        returns:
            dict w scores for different things
        """
        metrics = {
            'landing_height': 0.0,
            'surface_smoothness': 0.0,
            'well_contribution': 0.0,
            'overhang_created': 0.0
        }
        
        # find where piece went
        diff = grid - prev_grid
        if not np.any(diff):
            return metrics
            
        # see how high it landed
        for y in range(self.height):
            if np.any(diff[y] != 0):
                metrics['landing_height'] = self.height - y
                break
        
        # check if surface got smoother or bumpier
        _, heights_before = self._get_bumpiness_and_height(prev_grid)
        bumpiness_after, heights_after = self._get_bumpiness_and_height(grid)
        metrics['surface_smoothness'] = sum(abs(h1 - h2) for h1, h2 in zip(heights_before, heights_after))
        
        # see if it helped the well
        well_before, _ = self._analyze_well_structure(prev_grid)
        well_after, well_quality = self._analyze_well_structure(grid)
        metrics['well_contribution'] = well_quality if well_after and not well_before else 0.0
        
        # check if it made dangly bits
        overhangs_before = self._check_overhangs(prev_grid)
        overhangs_after = self._check_overhangs(grid)
        metrics['overhang_created'] = max(0, overhangs_after - overhangs_before)
        
        return metrics

    def _calculate_reward(self, prev_score: int, prev_lines: int) -> float:
        """
        figure out points for last move
        
        args:
            prev_score: points before move
            prev_lines: lines cleared before move
            
        returns:
            points earned
        """
        reward = 0.0
        
        # get current board state
        curr_grid = self.board.grid.astype(np.uint8)
        
        # points for clearing lines
        lines_cleared = self.board.lines_cleared - prev_lines
        if lines_cleared > 0:
            # base points for lines
            reward += lines_cleared * self.REWARD_WEIGHTS['lines_cleared']
            # bonus for tetris (4 lines)
            if lines_cleared == 4:
                reward += self.REWARD_WEIGHTS['tetris_bonus']
            # combo bonus
            self.combo_count += 1
            reward += self.combo_count * self.REWARD_WEIGHTS['combo_bonus']
        else:
            self.combo_count = 0
        
        # check board state
        holes = self._get_holes(curr_grid)
        bumpiness, heights = self._get_bumpiness_and_height(curr_grid)
        max_height = max(heights)
        
        # take away points for messy board
        reward += holes * self.REWARD_WEIGHTS['hole_penalty']
        reward += bumpiness * self.REWARD_WEIGHTS['bumpiness_penalty']
        reward += max_height * self.REWARD_WEIGHTS['height_penalty']
        
        # bonus for clearing whole board
        if np.all(curr_grid == 0):
            reward += self.REWARD_WEIGHTS['perfect_clear_bonus']
        
        # check well structure
        has_well, well_quality = self._analyze_well_structure(curr_grid)
        if has_well:
            reward += well_quality * self.REWARD_WEIGHTS['well_bonus']
        
        # points for flat bits
        flat_segments = self._detect_flat_surfaces(curr_grid)
        for _, length in flat_segments:
            reward += length * self.REWARD_WEIGHTS['flat_bonus']
        
        # check t-spin spots
        tspin_potential = self._evaluate_tspin_potential(curr_grid)
        reward += tspin_potential * self.REWARD_WEIGHTS['tspin_setup_bonus']
        
        # take away points for dangly bits
        overhangs = self._check_overhangs(curr_grid)
        reward += overhangs * self.REWARD_WEIGHTS['overhang_penalty']
        
        # big penalty if game over
        if self.board.game_over:
            reward += self.REWARD_WEIGHTS['game_over']
        
        # tiny penalty for each move to keep things quick
        reward += self.REWARD_WEIGHTS['move_penalty']
        
        return reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        """
        start fresh game
        
        args:
            seed: random seed to make things repeatable
            options: extra reset options (not used)
            
        returns:
            tuple w:
            - first board state
            - extra game info
        """
        super().reset(seed=seed)
        self.board = Board(self.width, self.height)
        obs, info = self._get_observation()
        
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(
                board_state=obs,
                score=info['score'],
                lines_cleared=info['lines_cleared'],
                next_piece=info['next_piece']
            )
            
        return cast(NDArray[np.uint8], obs), info

    def step(self, action: np.int64) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        """
        do a move in the game
        
        args:
            action: which move to do (check Action enum)
            
        returns:
            tuple w:
            - new board state
            - points earned
            - if game over
            - if episode cut short (not used in tetris)
            - extra info dict
        """
        if not isinstance(action, (int, np.integer)):
            raise TypeError(f"Action must be an integer, got {type(action)}")
        
        if not 0 <= action < len(Action):
            raise ValueError(f"Invalid action {action}. Must be between 0 and {len(Action)-1}")

        # save state before move
        prev_score = self.board.score
        prev_lines = self.board.lines_cleared
        
        # do the move
        action_enum = Action(int(action))
        self._execute_action(action_enum)
        
        # calc points
        reward = self._calculate_reward(prev_score, prev_lines)
        
        # get new state
        obs, info = self._get_observation()
        terminated = self.board.game_over
        truncated = False  # tetris doesnt use truncation
        
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(
                board_state=obs,
                score=info['score'],
                lines_cleared=info['lines_cleared'],
                next_piece=info['next_piece']
            )
        
        return cast(NDArray[np.uint8], obs), reward, terminated, truncated, info

    def _execute_action(self, action: Action) -> None:
        """
        do the move on the board
        
        args:
            action: which move to do
        """
        if action == Action.NOOP:
            _ = self.board.move_down()
        elif action == Action.LEFT:
            _ = self.board.move_left()
            _ = self.board.move_down()
        elif action == Action.RIGHT:
            _ = self.board.move_right()
            _ = self.board.move_down()
        elif action == Action.ROTATE_CW:
            _ = self.board.rotate(clockwise=True)
            _ = self.board.move_down()
        elif action == Action.ROTATE_CCW:
            _ = self.board.rotate(clockwise=False)
            _ = self.board.move_down()
        elif action == Action.SOFT_DROP:
            _ = self.board.move_down()
        elif action == Action.HARD_DROP:
            _ = self.board.hard_drop()

    def _get_observation(self) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        """
        get current game state
        
        returns:
            tuple w:
            - board state as numpy array
            - extra info dict
        """
        grid, current_piece, next_piece, score, lines_cleared, game_over = self.board.get_state()
        
        # make observation grid (includes current piece)
        obs = grid.copy().astype(np.uint8)  # make sure its uint8
        if current_piece:
            for x, y in current_piece.get_positions():
                if 0 <= y < self.height and 0 <= x < self.width:
                    obs[y][x] = ord(current_piece.type.value)
        
        info = {
            'score': score,
            'lines_cleared': lines_cleared,
            'game_over': game_over,
            'next_piece': next_piece.type.value if next_piece else None
        }
        
        return obs, info

    def render(self) -> NDArray[np.uint8] | None:
        """
        show current game state
        
        returns:
            numpy array: game state as 2d array for 'rgb_array' mode
            None: for 'human' mode since it shows directly
        """
        if self.render_mode == "rgb_array":
            obs, _ = self._get_observation()
            return cast(NDArray[np.uint8], obs)
        return None

    def close(self) -> None:
        """cleanup stuff"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None 