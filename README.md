# Tetris AI Testbed

A robust implementation of Tetris designed specifically as a standardized testbed for experimenting with various AI techniques. This project provides a clear API for AI agents to interact with the game, similar to OpenAI Gym environments.

## Features

- Full Tetris game engine implementation
- Standardized AI interface for experimentation
- Support for both visual and headless modes
- Comprehensive state representation and action space
- Performance optimized for rapid simulation
- Visualization tools for debugging

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tetris-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Manual Play Example

To try out the game manually:

```bash
cd examples
python manual_play.py
```

Controls:
- Left/Right Arrow: Move piece
- Up Arrow: Rotate clockwise
- Z: Rotate counterclockwise
- Down Arrow: Soft drop
- Space: Hard drop
- Q: Quit game

### AI Integration

The environment follows a similar interface to OpenAI Gym:

```python
from tetris import TetrisEnv, Action

# Create environment
env = TetrisEnv()

# Reset environment
obs, info = env.reset()

# Game loop
done = False
while not done:
    # Your AI agent's action selection here
    action = ...  # Choose an action from Action enum
    
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Optional: render the game state
    env.render()
    
    if terminated:
        break
```

## Project Structure

```
.
├── tetris/
│   ├── engine/         # Core game logic
│   ├── environment/    # AI environment interface
│   ├── visualization/  # Rendering and debugging tools
│   └── utils/         # Helper functions
├── tests/             # Test suite
├── examples/          # Example implementations
└── docs/             # Documentation
```

## State Representation

The game state is represented as a 2D numpy array where:
- 0: Empty cell
- ASCII values of piece types ('I', 'O', 'T', etc.): Filled cells

Additional information available in the info dictionary:
- score: Current game score
- lines_cleared: Number of lines cleared
- next_piece: Type of the next piece
- game_over: Whether the game has ended

## Action Space

Available actions:
- NOOP (0): Let piece fall naturally
- LEFT (1): Move piece left
- RIGHT (2): Move piece right
- ROTATE_CW (3): Rotate piece clockwise
- ROTATE_CCW (4): Rotate piece counterclockwise
- SOFT_DROP (5): Move piece down faster
- HARD_DROP (6): Drop piece to bottom instantly

## Reward Structure

The reward function considers:
- Points from clearing lines
- Bonus for clearing multiple lines simultaneously
- Penalty for game over

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details 