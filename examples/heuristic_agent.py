"""
heuristic-based tetris agent w testing framework

this agent uses simulation to check all possible final positions for each piece
and picks the best move based on heuristic evaluation
"""
import time
import numpy as np
import copy
from typing import Callable 
from dataclasses import dataclass

# import tetris env
from tetris import TetrisEnv, Action
from tetris.engine.board import Board


@dataclass
class MoveEvaluation:
    """evaluation of a potential move"""
    action_sequence: list[Action]  # sequence of actions to reach this position
    score: float  # heuristic score
    metrics: dict[str, float]  # detailed metrics for analysis


class HeuristicAgent:
    """
    a heuristic-based agent for playing tetris
    
    this agent simulates all possible final positions for the current piece
    and picks the best move based on heuristic evaluation
    """
    
    def __init__(self, weights: dict[str, float] | None = None, debug: bool = False):
        """
        init the agent w heuristic weights
        
        args:
            weights: dict of weights for different heuristics
            debug: whether to print debug info
        """
        # default weights if none provided
        self.weights: dict[str, float] = weights or {
            'holes': -4.0,           # penalty for making holes
            'height': -0.5,          # penalty for increasing height
            'bumpiness': -1.0,       # penalty for uneven surface
            'lines_cleared': 3.0,    # reward for clearing lines
            'well_depth': 0.5,       # reward for making wells for I pieces
        }
        self.debug: bool = debug
    
    def get_best_action(self, env: TetrisEnv) -> tuple[Action, float, dict[str, float]]:
        """
        figure out the best action to take 
        
        args:
            env: the tetris env
            
        returns:
            tuple of (best action, decision time in ms, metrics)
        """
        start_time = time.time()
        
        # get all possible final positions
        evaluations = self._evaluate_all_positions(env)
        
        if not evaluations:
            # if no valid moves just do a hard drop
            if self.debug:
                print("no valid moves found, using HARD_DROP")
            return Action.HARD_DROP, (time.time() - start_time) * 1000, {}
        
        # find the best evaluation
        best_eval = max(evaluations, key=lambda e: e.score)
        
        if self.debug:
            print(f"best action sequence: {[a.name for a in best_eval.action_sequence]}")
            print(f"best score: {best_eval.score}")
            print(f"metrics: {best_eval.metrics}")
            print(f"total evaluations: {len(evaluations)}")
            
            # print top 3 evaluations
            sorted_evals = sorted(evaluations, key=lambda e: e.score, reverse=True)[:3]
            for i, eval in enumerate(sorted_evals):
                print(f"evaluation {i+1}:")
                print(f"  action sequence: {[a.name for a in eval.action_sequence]}")
                print(f"  score: {eval.score}")
                print(f"  metrics: {eval.metrics}")
        
        # return first action in best sequence
        decision_time_ms = (time.time() - start_time) * 1000
        return best_eval.action_sequence[0], decision_time_ms, best_eval.metrics
    
    def _evaluate_all_positions(self, env: TetrisEnv) -> list[MoveEvaluation]:
        """
        check out all possible final positions for current piece
        
        args:
            env: the tetris env
            
        returns:
            list of move evaluations
        """
        # make a copy of env for simulation
        sim_env = copy.deepcopy(env)
        
        # list to store all evaluations
        evaluations: list[MoveEvaluation] = []
        
        # track visited states to avoid duplicates
        visited_states: set[int] = set()
        
        # queue for bfs traversal of possible moves
        # each entry is (action_sequence, board_state)
        queue: list[tuple[list[Action], Board]] = [([], sim_env.board)]
        
        # limit num of states to explore to avoid infinite loops
        max_states = 1000
        states_explored = 0
        
        # track action counts for debugging
        action_counts = {action.name: 0 for action in Action}
        
        # track errors for debugging
        error_counts = {action.name: 0 for action in Action}
        
        while queue and states_explored < max_states:
            action_sequence, current_board = queue.pop(0)
            states_explored += 1
            
            if self.debug:
                print(f"exploring state {states_explored} w action sequence: {[a.name for a in action_sequence]}")
            
            # skip if we've been here b4
            board_hash = self._hash_board_state(current_board.grid, current_board)
            if board_hash in visited_states:
                if self.debug:
                    print("  skipping already visited state")
                continue
            
            visited_states.add(board_hash)

            action_funcs: list[tuple[Action, Callable[[Board], bool | None]]] = [
                (Action.NOOP, lambda b: True),      
                (Action.LEFT, lambda b: b.move_left()),
                (Action.RIGHT, lambda b: b.move_right()),
                (Action.ROTATE_CW, lambda b: b.rotate(clockwise=True)),
                (Action.ROTATE_CCW, lambda b: b.rotate(clockwise=False)),
                (Action.SOFT_DROP, lambda b: b.move_down())
            ]
            
            # try each possible action
            for action, action_func in action_funcs:
                # make a copy of the board
                next_board = current_board.copy()
                
                # do the action
                try:
                    if self.debug:
                        print(f"  trying action: {action.name}")
                    
                    # use the board method directly
                    result = action_func(next_board)
                    
                    # skip if action did nothing
                    if not result:
                        if self.debug:
                            print(f"    action {action.name} did nothing, skipping")
                        continue
                    
                    # skip if game over
                    if next_board.game_over:
                        if self.debug:
                            print("    game over, skipping")
                        continue
                    
                    # skip if we've seen this state b4
                    next_hash = self._hash_board_state(next_board.grid, next_board)
                    if next_hash == board_hash:
                        if self.debug:
                            print("    state didnt change, skipping")
                        continue
                    
                    # add to queue for more exploring
                    queue.append((action_sequence + [action], next_board))
                    action_counts[action.name] += 1
                    
                    if self.debug:
                        print(f"    added to queue w new sequence: {[a.name for a in action_sequence + [action]]}")
                    
                    # if this was a soft drop check if its a good landing spot
                    if action == Action.SOFT_DROP:
                        # see if piece would land on next move down
                        test_board = next_board.copy()
                        if not test_board.move_down():  # piece would land next
                            # make new board for final state
                            final_board = next_board.copy()
                            # hard drop to place piece
                            final_board.hard_drop()
                            
                            # check out this final position
                            info = {
                                'score': final_board.score,
                                'lines_cleared': final_board.lines_cleared,
                                'prev_lines_cleared': current_board.lines_cleared
                            }
                            metrics = self._evaluate_position(final_board.grid, info)
                            score = self._calculate_score(metrics)
                            
                            # add to evaluations
                            evaluations.append(MoveEvaluation(
                                action_sequence=action_sequence + [action],
                                score=score,
                                metrics=metrics
                            ))
                            
                            if self.debug:
                                print(f"    piece would land after SOFT_DROP, added eval w score {score}")
                except Exception as e:
                    error_counts[action.name] += 1
                    if self.debug:
                        print(f"    error doing action {action.name}: {e}")
                    continue
            
            # try hard drop to get final position
            try:
                if self.debug:
                    print("  trying HARD_DROP")
                
                # make copy of board for final state
                final_board = current_board.copy()
                
                # save state b4 hard drop
                prev_state = final_board.grid.copy()
                
                # do the hard drop
                final_board.hard_drop()
                
                # if piece was placed (board changed)
                if not np.array_equal(final_board.grid, prev_state):
                    # check out this final position
                    info = {
                        'score': final_board.score,
                        'lines_cleared': final_board.lines_cleared,
                        'prev_lines_cleared': current_board.lines_cleared
                    }
                    metrics = self._evaluate_position(final_board.grid, info)
                    score = self._calculate_score(metrics)
                    
                    # add to evaluations
                    evaluations.append(MoveEvaluation(
                        action_sequence=action_sequence + [Action.HARD_DROP],
                        score=score,
                        metrics=metrics
                    ))
                    action_counts[Action.HARD_DROP.name] += 1
                    
                    if self.debug:
                        print(f"    added eval w score {score}")
                else:
                    if self.debug:
                        print("    HARD_DROP did nothing, skipping")
            except Exception as e:
                error_counts[Action.HARD_DROP.name] += 1
                if self.debug:
                    print(f"    error doing HARD_DROP: {e}")
        
        if self.debug:
            print(f"explored {states_explored} states, found {len(evaluations)} valid final positions")
            print(f"action counts during exploring: {action_counts}")
            print(f"error counts during exploring: {error_counts}")
        
        return evaluations
    
    def _hash_board_state(self, board: np.ndarray[tuple[int, ...], np.dtype[np.int8]], current_board: Board | None = None) -> int:
        """
        make a hash of board state including current piece pos and shape
        
        args:
            board: the board grid
            current_board: the board obj w current piece
            
        returns:
            hash value for the state
        """
        # if we got a board obj w current piece include its pos and shape in hash
        if current_board and current_board.current_piece:
            piece = current_board.current_piece
            # combine board state w piece info
            piece_info = f"{piece.x},{piece.y},{piece.rotation}"
            return hash(board.tobytes() + piece_info.encode())
        else:
            # just use board grid if no piece
            return hash(board.tobytes())
    
    def _evaluate_position(self, 
                           new_board: np.ndarray[tuple[int, ...], np.dtype[np.int8]], 
                           info: dict[str, int]) -> dict[str, float]:
        """
        check how good a board position is using diff heuristics
        
        args:
            new_board: the board after move
            info: extra info from env
            
        returns:
            dict of metrics
        """
        # count holes (empty cells w filled cells above)
        holes = self._count_holes(new_board)
        
        # calc height and bumpiness
        bumpiness, heights = self._get_bumpiness_and_heights(new_board)
        max_height = max(heights) if heights else 0
        
        # count lines cleared
        lines_cleared: int = info.get('lines_cleared', 0) - info.get('prev_lines_cleared', 0)
        
        # look for wells (cols w higher cols next to them)
        well_depth = self._calculate_well_depth(heights)
        
        return {
            'holes': float(holes),
            'height': float(max_height),
            'bumpiness': float(bumpiness),
            'lines_cleared': float(lines_cleared),
            'well_depth': float(well_depth)
        }
    
    def _calculate_score(self, metrics: dict[str, float]) -> float:
        """
        calc overall score from metrics and weights
        
        args:
            metrics: dict of metrics
            
        returns:
            overall score
        """
        return sum(metrics[key] * self.weights[key] for key in self.weights if key in metrics)
    
    def _count_holes(self, board: np.ndarray[tuple[int, ...], np.dtype[np.int8]]) -> int:
        """
        count holes in the board
        
        a hole is an empty cell w filled cells above it
        
        args:
            board: the board state
            
        returns:
            num of holes
        """
        holes = 0
        # for each col
        for col in range(board.shape[1]):
            # find highest filled cell
            for row in range(board.shape[0]):
                if board[row, col] > 0:  # found filled cell
                    # count empty cells below it
                    for r in range(row + 1, board.shape[0]):
                        if board[r, col] == 0:
                            holes += 1
                    break
        return holes
    
    def _get_bumpiness_and_heights(self, board: np.ndarray[tuple[int, ...], np.dtype[np.int8]]) -> tuple[float, list[int]]:
        """
        calc how bumpy board is and heights of cols
        
        bumpiness is sum of height diffs between cols next to each other
        
        args:
            board: the board state
            
        returns:
            tuple of (bumpiness, list of col heights)
        """
        heights: list[int] = []
        # for each col
        for col in range(board.shape[1]):
            # find highest filled cell
            for row in range(board.shape[0]):
                if board[row, col] > 0:
                    heights.append(board.shape[0] - row)
                    break
            else:
                heights.append(0)  # empty col
        
        # calc bumpiness
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness, heights
    
    def _calculate_well_depth(self, heights: list[int]) -> float:
        """
        calc how deep wells are in board
        
        a well is a col w cols next to it at least 2 blocks higher
        
        args:
            heights: list of col heights
            
        returns:
            total well depth
        """
        well_depth = 0
        for i in range(len(heights)):
            # check if this col is a well
            left_higher = i > 0 and heights[i-1] - heights[i] >= 2
            right_higher = i < len(heights)-1 and heights[i+1] - heights[i] >= 2
            
            if left_higher and right_higher:
                # deep well (both sides higher)
                well_depth += min(heights[i-1], heights[i+1]) - heights[i]
            elif left_higher:
                well_depth += heights[i-1] - heights[i]
            elif right_higher:
                well_depth += heights[i+1] - heights[i]
        
        return well_depth 