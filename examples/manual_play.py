import sys
import pygame
import time
from typing import Any
import numpy as np

# add parent dir to python path so we can import our tetris pkg
sys.path.append('..')

from tetris.environment.tetris_env import TetrisEnv, Action
from tetris.visualization.renderer import TetrisRenderer


def main():
    """lets u play tetris manually w keyboard controls"""
    # init env and renderer
    env = TetrisEnv()
    renderer = TetrisRenderer()
    
    # key mappings
    key_action_map = {
        pygame.K_LEFT: Action.LEFT,
        pygame.K_RIGHT: Action.RIGHT,
        pygame.K_UP: Action.ROTATE_CW,
        pygame.K_z: Action.ROTATE_CCW,
        pygame.K_DOWN: Action.SOFT_DROP,
        pygame.K_SPACE: Action.HARD_DROP,
    }
    
    # game loop setup
    clock = pygame.time.Clock()
    running = True
    auto_drop_time = time.time()
    auto_drop_delay = 1.0  # secs between auto drops
    
    # get starting state
    obs, info = env.reset()
    terminated = False
    
    while running:
        current_time = time.time()
        
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in key_action_map:
                    action = key_action_map[event.key]
                    obs, reward, terminated, truncated, info = env.step(action.value)
                    auto_drop_time = current_time  # reset auto-drop timer
        
        # auto-drop piece if time is up
        if current_time - auto_drop_time >= auto_drop_delay:
            obs, reward, terminated, truncated, info = env.step(Action.NOOP.value)
            auto_drop_time = current_time
        
        # show current state
        renderer.render(
            board_state=obs,
            score=info['score'],
            lines_cleared=info['lines_cleared'],
            next_piece=info['next_piece']
        )
        
        # check if game over
        if terminated:
            print(f"game over! final score: {info['score']}")
            print(f"lines cleared: {info['lines_cleared']}")
            running = False
        
        # control game speed
        clock.tick(60)
    
    # cleanup
    renderer.close()


if __name__ == "__main__":
    main() 