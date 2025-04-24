# Debugging Journey: Fixing the Heuristic Tetris Agent

## Evolution of Our Approach

### Initial Simple Heuristics

When we first started developing the Tetris agent, we began with a simple approach focused on avoiding gaps (holes) in the board. The agent would prioritize moves that minimized the number of holes created.

This simple strategy worked to some extent but had significant limitations. The agent would often create tall, unstable structures that would eventually lead to game over.

### Adding Height Considerations

To address the issue of tall, unstable structures, we added a height penalty to our heuristic evaluation. The agent would now consider both holes and the maximum height of the board when making decisions.

This improved performance, but the agent still struggled with creating flat surfaces and efficiently clearing lines.

### Expanding the Heuristic Set

We continued to refine our approach by adding more heuristics:

1. **Bumpiness**: Penalizing uneven surfaces to encourage flat board states
2. **Lines Cleared**: Rewarding moves that clear lines
3. **Well Depth**: Rewarding the creation of wells that could accommodate I-pieces for Tetris clears

These additional heuristics gave the agent a more nuanced understanding of good board states, but we still needed a way for the agent to effectively explore different positions.

### End State Simulation

The final evolution of our approach was to implement end state simulation. Instead of evaluating just the immediate result of an action, the agent would simulate all possible sequences of actions and evaluate the final board states.

This approach allowed the agent to consider complex action sequences like "move left twice, rotate clockwise, then hard drop" and choose the one that resulted in the best board state according to our heuristics.

## Initial Problem

After implementing the end state simulation approach, we encountered a critical issue: the heuristic agent was failing to explore different positions for Tetris pieces. When running the test script, we observed that the agent was only executing `HARD_DROP` actions, resulting in poor performance.

## Investigation Process

### Step 1: Adding Debug Output

We first added debug output to the `test_heuristic_agent.py` file to understand what was happening during the agent's decision-making process. This included:

- Printing the initial board state
- Displaying the current piece type, position, and shape
- Testing manual movements (left, right, rotate) to verify if they worked correctly
- Showing the state before and after each action

The debug output revealed a critical issue: while manual movements worked correctly in the test script, they failed during the agent's state exploration with the message "State didn't change, skipping."

### Step 2: Examining the Code

We examined the following files to understand the issue:

1. `examples/heuristic_agent.py` - The main agent implementation
2. `examples/test_heuristic_agent.py` - The testing framework
3. `tetris/engine/board.py` - The Tetris board implementation

The key issue was identified in the `_evaluate_all_positions` method of the `HeuristicAgent` class. The agent was using a hash of the board state to detect duplicates, but this hash didn't include the current piece's position and rotation. As a result, moving the piece left/right or rotating it didn't change the hash, causing the agent to skip these states.

## Failed Attempts

### Attempt 1: Using Board Methods Directly

Our first attempt was to modify the agent to use board methods directly for exploration instead of the environment's step method. This included:

```python
# Try each possible action
for action, action_func in [
    (Action.LEFT, lambda b: b.move_left()),
    (Action.RIGHT, lambda b: b.move_right()),
    (Action.ROTATE_CW, lambda b: b.rotate(clockwise=True)),
    (Action.ROTATE_CCW, lambda b: b.rotate(clockwise=False)),
    (Action.SOFT_DROP, lambda b: b.move_down())
]:
    # Create a copy of the board
    next_board = current_board.copy()
    
    # Execute the action
    try:
        if self.debug:
            print(f"  Trying action: {action.name}")
        
        # Use the board method directly
        result = action_func(next_board)
        
        # Skip if the action didn't change the state
        if not result:
            if self.debug:
                print(f"    Action {action.name} had no effect, skipping")
            continue
```

However, this didn't solve the issue because the hash function still didn't account for the piece's position and rotation.

### Attempt 2: Breadth-First Search Approach

We implemented a breadth-first search (BFS) approach for move exploration, tracking visited states to avoid duplicates:

```python
# Queue for BFS traversal of possible moves
# Each entry is (action_sequence, board_state)
queue = [([], sim_env.board)]

# Limit the number of states to explore to avoid infinite loops
max_states = 1000
states_explored = 0
```

This approach was on the right track but still failed due to the hash function issue.

## Successful Solution

The key fix was modifying the `_hash_board_state` method to include the current piece's position and rotation in the hash:

```python
def _hash_board_state(self, board: np.ndarray[Any, np.dtype[np.int8]], current_board=None) -> int:
    """
    Create a hash of the board state including the current piece position and shape.
    
    Args:
        board: The board grid
        current_board: The Board object containing the current piece
        
    Returns:
        Hash value representing the state
    """
    # If we have a board object with a current piece, include its position and shape in the hash
    if current_board and current_board.current_piece:
        piece = current_board.current_piece
        # Combine the board state with piece information
        piece_info = f"{piece.x},{piece.y},{piece.rotation}"
        return hash(board.tobytes() + piece_info.encode())
    else:
        # Fall back to just the board grid
        return hash(board.tobytes())
```

This change ensured that moving the piece left/right or rotating it would result in a different hash, allowing the agent to properly explore different positions.

## Results

After implementing the fix, the agent successfully explored many different positions and actions. The debug output showed that it was now trying various action sequences, including:

- Moving pieces left and right
- Rotating pieces
- Performing soft drops and hard drops

The agent was now able to evaluate multiple final positions and select the best move based on the heuristic evaluation.

## Parameter Tuning

With the agent now properly exploring different positions, we experimented with different weight combinations for our heuristics:

1. **Holes**: We tried values ranging from -5.0 to -2.0, with -4.0 providing a good balance
2. **Height**: Values from -1.0 to -0.1, with -0.5 working well
3. **Bumpiness**: Values from -2.0 to -0.5, with -1.0 being effective
4. **Lines Cleared**: Values from 2.0 to 4.0, with 3.0 providing good results
5. **Well Depth**: Values from 0.0 to 1.0, with 0.5 being a good choice

The final weights we settled on were:
- Holes: -4.0
- Height: -0.5
- Bumpiness: -1.0
- Lines Cleared: 3.0
- Well Depth: 0.5

These weights resulted in an agent that could clear an average of 103 lines per episode and achieve scores up to 26,300.

## Lessons Learned

1. **State Representation Matters**: When implementing state-based algorithms, it's crucial to ensure that the state representation (in this case, the hash) captures all relevant aspects of the state.

2. **Debug Output is Essential**: Adding detailed debug output was key to identifying the issue. It allowed us to see exactly what was happening during the agent's decision-making process.

3. **Test Manual Actions**: Testing manual actions separately from the agent's exploration helped identify that the issue was with the state tracking, not with the actions themselves.

4. **Breadth-First Search for Exploration**: Using BFS for exploring possible moves is an effective approach for finding the best action sequence in a game like Tetris, its not perfect but it works.

5. **Balanced Heuristics**: A combination of different heuristics, properly weighted, is necessary for good performance. No single heuristic is sufficient on its own.

6. **Simulation-Based Approach**: End state simulation, while computationally expensive, provides much better results than simple one-step lookahead.

## Things we could do better

1. **Optimize State Exploration**: The current implementation explores a large number of states, which can be computationally expensive. Optimizing the exploration strategy could improve performance.

2. **Tune Heuristic Weights**: The current weights were chosen based on limited testing. A more systematic approach to weight tuning could lead to better performance.

3. **Add More Heuristics**: Additional heuristics, such as rewarding T-spins or penalizing overhangs, could improve the agent's decision-making.

4. **Implement Lookahead**: Currently, the agent only considers the current piece. Implementing lookahead to consider the next piece as well could lead to better decisions. 