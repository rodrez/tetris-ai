"""
testing framework for the heuristic tetris agent

this script lets u run tests with different heuristic weights and
outputs detailed results to the terminal for analysis
"""
import sys
import time
import os
import argparse
from argparse import Namespace
import numpy as np
from numpy.typing import NDArray
import json
from typing import Any, override, cast, Protocol, TypedDict
from datetime import datetime
try:
    from tabulate import tabulate
except ImportError:
    print("tabulate package not found. Please install it with: pip install tabulate")
    sys.exit(1)

# add the parent dir to python path
sys.path.append('..')

from tetris import TetrisEnv, Action, TetrisRenderer
from examples.heuristic_agent import HeuristicAgent

# define a protocol for tetrisrenderer to help w type checking
class TetrisRendererProtocol(Protocol):
    def render(self, board_state: NDArray[np.uint8], score: int, lines_cleared: int, next_piece: str | None = None) -> None: ...
    def close(self) -> None: ...

# custom tetrisenv wrapper that doesnt auto move down after each action
class CustomTetrisEnv(TetrisEnv):
    """
    a custom wrapper for tetrisenv that doesnt automatically move the piece down
    after each action. lets the agent explore different positions better.
    """
    
    @override
    def _execute_action(self, action: Action) -> None:
        """
        execute the given action on game board without auto downward movement.
        
        args:
            action: the action to execute
        """
        if action == Action.NOOP:
            pass  # do nothing
        elif action == Action.LEFT:
            _ = self.board.move_left()
        elif action == Action.RIGHT:
            _ = self.board.move_right()
        elif action == Action.ROTATE_CW:
            _ = self.board.rotate(clockwise=True)
        elif action == Action.ROTATE_CCW:
            _ = self.board.rotate(clockwise=False)
        elif action == Action.SOFT_DROP:
            _ = self.board.move_down()
        elif action == Action.HARD_DROP:
            _ = self.board.hard_drop()

# dir for storing test results
RESULTS_DIR = "agent_test_results"

# define a typeddict for episode results
class EpisodeResult(TypedDict):
    score: int
    lines_cleared: int
    steps: int
    total_reward: float
    duration: float
    metrics_history: list[dict[str, float | int | str]]
    action_counts: dict[str, int]

# define a typeddict for aggregated results
class AggregatedResults(TypedDict):
    weights: dict[str, float]
    episodes: int
    avg_score: float
    avg_lines: float
    avg_steps: float
    avg_duration: float
    max_score: int
    max_lines: int
    episode_results: list[EpisodeResult]
    action_counts: dict[str, int]

def run_test_episode(agent: HeuristicAgent, 
                    delay: float = 0.01, 
                    render: bool = True,
                    verbose: bool = False,
                    use_custom_env: bool = False,
                    debug: bool = False) -> EpisodeResult:
    """
    run a single test episode w the agent.
    
    Args:
        agent: the tetris agent to test
        delay: delay between moves in secs
        render: whether to render the game
        verbose: whether to print detailed move info
        use_custom_env: whether to use custom env
        debug: whether to enable debug mode
        
    returns:
        dictionary with episode results
    """
    # init env and renderer
    env = CustomTetrisEnv() if use_custom_env else TetrisEnv()
    renderer: TetrisRendererProtocol | None = TetrisRenderer() if render else None
    
    # reset env
    obs, info = env.reset()
    
    # debug: print initial state
    if debug:
        print("\ninitial board state:")
        print(obs)
        if env.board.current_piece:
            print(f"current piece: {env.board.current_piece.type.value}")
            print(f"current piece pos: ({env.board.current_piece.x}, {env.board.current_piece.y})")
            # cast shape to ndarray to handle type checking
            shape_array = cast(NDArray[np.int8], env.board.current_piece.shape)
            print(f"current piece shape:\n{shape_array}")
            
            # try moving piece manually to see if it works
            print("\ntrying manual movements:")
            # try left
            left_result = env.board.move_left()
            print(f"move left result: {left_result}")
            if left_result:
                print(f"new pos after left: ({env.board.current_piece.x}, {env.board.current_piece.y})")
            
            # try right
            right_result = env.board.move_right()
            print(f"move right result: {right_result}")
            if right_result:
                print(f"new pos after right: ({env.board.current_piece.x}, {env.board.current_piece.y})")
            
            # try rotate
            rotate_result = env.board.rotate(clockwise=True)
            print(f"rotate cw result: {rotate_result}")
            if rotate_result:
                # cast shape to ndarray to handle type checking
                shape_array = cast(NDArray[np.int8], env.board.current_piece.shape)
                print(f"new shape after rotate:\n{shape_array}")
                
            # reset piece pos and rotation for actual test
            _, _ = env.reset()
    
    total_reward = 0
    steps = 0
    start_time = time.time()
    
    # metrics tracking
    metrics_history: list[dict[str, float | int | str]] = []
    action_counts = {action.name: 0 for action in Action}
    
    # episode loop
    while True:
        # get best action from agent
        best_action, decision_time_ms, metrics = agent.get_best_action(env)
        
        # debug: print current state b4 taking action
        if debug:
            print(f"\nstep {steps + 1} - before action {best_action.name}:")
            if env.board.current_piece:
                print(f"current piece: {env.board.current_piece.type.value}")
                print(f"current piece pos: ({env.board.current_piece.x}, {env.board.current_piece.y})")
                # cast shape to ndarray to handle type checking
                shape_array = cast(NDArray[np.int8], env.board.current_piece.shape)
                print(f"current piece shape:\n{shape_array}")
        
        # track action counts
        action_counts[best_action.name] = action_counts.get(best_action.name, 0) + 1
        
        # print move details if verbose
        if verbose:
            print(f"\nstep {steps + 1}:")
            print(f"  action: {best_action.name}")
            print(f"  decision time: {decision_time_ms:.2f} ms")
            print(f"  metrics: {metrics}")
        
        # take the action
        obs, reward, terminated, _, info = env.step(np.int64(best_action.value))
        total_reward += reward
        steps += 1
        
        # debug: print state after taking action
        if debug:
            print(f"after action {best_action.name}:")
            print(obs)
            if env.board.current_piece:
                print(f"current piece: {env.board.current_piece.type.value}")
                print(f"current piece pos: ({env.board.current_piece.x}, {env.board.current_piece.y})")
                # cast shape to ndarray to handle type checking
                shape_array = cast(NDArray[np.int8], env.board.current_piece.shape)
                print(f"current piece shape:\n{shape_array}")
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
        
        # store metrics for this step
        step_data = {
            'step': steps,
            'action': best_action.name,
            'reward': float(reward),
            'score': info['score'],
            'lines_cleared': info['lines_cleared'],
            'decision_time_ms': decision_time_ms,
            **metrics
        }
        metrics_history.append(step_data)
        
        # render current state
        if renderer:
            # cast board_state to ndarray to handle type checking
            board_state_array = cast(NDArray[np.uint8], obs)
            # use renderer to show current state
            renderer.render(
                board_state=board_state_array,
                score=info['score'],
                lines_cleared=info['lines_cleared'],
                next_piece=info['next_piece']
            )
        
        # add delay to make it viewable
        time.sleep(delay)
        
        # check if episode is done
        if terminated:
            break
    
    # calc episode duration
    duration = time.time() - start_time
    
    # close renderer
    if renderer:
        renderer.close()
    
    # return episode results
    return {
        'score': info['score'],
        'lines_cleared': info['lines_cleared'],
        'steps': steps,
        'total_reward': total_reward,
        'duration': duration,
        'metrics_history': metrics_history,
        'action_counts': action_counts
    }


def run_weight_test(weights: dict[str, float], 
                   episodes: int = 3, 
                   delay: float = 0.01,
                   render: bool = True,
                   verbose: bool = False,
                   debug: bool = False,
                   use_custom_env: bool = False) -> AggregatedResults:
    """
    run multiple episodes w given weights and return aggregated results
    
    args:
        weights: dict of weights for heuristic agent
        episodes: num of episodes to run
        delay: delay between moves in secs
        render: whether to render game
        verbose: whether to print detailed move info
        debug: whether to enable debug mode
        use_custom_env: whether to use custom env
        
    returns:
        dict with aggregated results
    """
    # create agent w given weights
    agent = HeuristicAgent(weights=weights, debug=debug)
    
    # track results across episodes
    all_results: list[EpisodeResult] = []
    
    for episode in range(episodes):
        print(f"\n--- episode {episode + 1}/{episodes} ---")
        
        # run episode
        result = run_test_episode(
            agent=agent,
            delay=delay,
            render=render,
            verbose=verbose,
            use_custom_env=use_custom_env,
            debug=debug
        )
        
        # print episode summary
        print(f"\nepisode {episode + 1} summary:")
        print(f"  score: {result['score']}")
        print(f"  lines cleared: {result['lines_cleared']}")
        print(f"  steps: {result['steps']}")
        print(f"  duration: {result['duration']:.2f} seconds")
        
        # store results
        all_results.append(result)
    
    # aggregate action counts
    action_counts = {action.name: 0 for action in Action}
    for result in all_results:
        for action, count in result['action_counts'].items():
            action_counts[action] = action_counts.get(action, 0) + count
    
    # create and return aggregated results
    return AggregatedResults(
        weights=weights,
        episodes=episodes,
        avg_score=sum(r['score'] for r in all_results) / episodes,
        avg_lines=sum(r['lines_cleared'] for r in all_results) / episodes,
        avg_steps=sum(r['steps'] for r in all_results) / episodes,
        avg_duration=sum(r['duration'] for r in all_results) / episodes,
        max_score=max(r['score'] for r in all_results),
        max_lines=max(r['lines_cleared'] for r in all_results),
        episode_results=all_results,
        action_counts=action_counts
    )


def save_results(results: AggregatedResults, test_name: str) -> str:
    """
    save test results to a file
    
    args:
        results: test results to save
        test_name: name of the test
        
    returns:
        path to saved file
    """
    # create results dir if it doesnt exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # generate filename w timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # save results as json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nresults saved to {filepath}")
    return filepath


def print_results_table(results: AggregatedResults) -> None:
    """
    print a formatted table of test results.
    
    args:
        results: test results to print
    """
    # print weights
    print("\n=== weights ===")
    weights_table = [[weight, value] for weight, value in results['weights'].items()]
    print(tabulate(weights_table, headers=["weight", "value"], tablefmt="grid"))
    
    # print aggregated metrics
    print("\n=== aggregated metrics ===")
    metrics_table = [
        ["avg score", f"{results['avg_score']:.2f}"],
        ["avg lines cleared", f"{results['avg_lines']:.2f}"],
        ["avg steps", f"{results['avg_steps']:.2f}"],
        ["avg duration (s)", f"{results['avg_duration']:.2f}"],
        ["max score", results['max_score']],
        ["max lines cleared", results['max_lines']]
    ]
    print(tabulate(metrics_table, headers=["metric", "value"], tablefmt="grid"))
    
    # print action counts
    print("\n=== action counts ===")
    action_table = [[action, count] for action, count in results['action_counts'].items()]
    print(tabulate(action_table, headers=["action", "count"], tablefmt="grid"))
    
    # print episode results
    print("\n=== episode results ===")
    episode_table = [
        [i+1, r['score'], r['lines_cleared'], r['steps'], f"{r['duration']:.2f}"]
        for i, r in enumerate(results['episode_results'])
    ]
    print(tabulate(
        episode_table, 
        headers=["episode", "score", "lines", "steps", "duration (s)"],
        tablefmt="grid"
    ))


def main():
    """main function to run the testing framework."""
    # parse command line args
    parser = argparse.ArgumentParser(description='test the heuristic tetris agent')
    _ = parser.add_argument('--episodes', type=int, default=3, help='num of episodes to run')
    _ = parser.add_argument('--delay', type=float, default=0.01, help='delay between moves in secs')
    _ = parser.add_argument('--no-render', action='store_true', help='disable rendering')
    _ = parser.add_argument('--verbose', action='store_true', help='print detailed move info')
    _ = parser.add_argument('--debug', action='store_true', help='enable debug mode for agent')
    _ = parser.add_argument('--test-name', type=str, default='heuristic_test', help='name for the test')
    _ = parser.add_argument('--custom-env', action='store_true', help='use custom env without auto downward movement')
    
    # weight params
    _ = parser.add_argument('--holes-weight', type=float, default=-4.0, help='weight for holes')
    _ = parser.add_argument('--height-weight', type=float, default=-0.5, help='weight for height')
    _ = parser.add_argument('--bumpiness-weight', type=float, default=-1.0, help='weight for bumpiness')
    _ = parser.add_argument('--lines-weight', type=float, default=3.0, help='weight for lines cleared')
    _ = parser.add_argument('--well-weight', type=float, default=0.5, help='weight for well depth')
    
    args: Namespace = parser.parse_args()
    
    # create weights dict from args
    weights: dict[str, float] = {
        'holes': args.holes_weight,
        'height': args.height_weight,
        'bumpiness': args.bumpiness_weight,
        'lines_cleared': args.lines_weight,
        'well_depth': args.well_weight
    }
    
    print(f"\nrunning {args.episodes} episodes w these weights:")
    for weight, value in weights.items():
        print(f"  {weight}: {value}")
    
    # run test w specified weights
    results = run_weight_test(
        weights=weights,
        episodes=args.episodes,
        delay=args.delay,
        render=not args.no_render,
        verbose=args.verbose,
        debug=args.debug,
        use_custom_env=args.custom_env
    )
    
    # print results table
    print_results_table(results)
    
    # save results
    save_results(results, args.test_name)


if __name__ == "__main__":
    main() 