"""
pygame stuff to show the tetris game
"""
import pygame
import numpy as np
import time


class TetrisRenderer:
    """
    draws the tetris game using pygame w some cool effects
    """

    # colors for the pieces (r,g,b) - main colors + darker shades for 3d look
    COLORS: dict[int, tuple[int, int, int]] = {
        0: (25, 25, 35),        # dark bg
        ord('I'): (66, 197, 245),    # cool cyan
        ord('O'): (245, 210, 66),    # sunny yellow
        ord('T'): (147, 88, 247),    # funky purple
        ord('S'): (98, 245, 112),    # fresh green
        ord('Z'): (245, 88, 88),     # hot red
        ord('J'): (88, 101, 242),    # deep blue
        ord('L'): (245, 150, 66),    # juicy orange
    }

    # darker colors for 3d shadows
    SHADE_COLORS: dict[int, tuple[int, int, int]] = {
        key: (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40))
        for key, color in COLORS.items()
    }

    # brighter colors for shiny bits
    HIGHLIGHT_COLORS: dict[int, tuple[int, int, int]] = {
        key: (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40))
        for key, color in COLORS.items()
    }

    # what each piece looks like in preview 
    PREVIEW_SHAPES: dict[str, list[list[int]]] = {
        'I': [[1, 1, 1, 1]],
        'O': [[1, 1],
              [1, 1]],
        'T': [[0, 1, 0],
              [1, 1, 1]],
        'S': [[0, 1, 1],
              [1, 1, 0]],
        'Z': [[1, 1, 0],
              [0, 1, 1]],
        'J': [[1, 0, 0],
              [1, 1, 1]],
        'L': [[0, 0, 1],
              [1, 1, 1]]
    }

    def __init__(self, 
                 width: int = 10, 
                 height: int = 20, 
                 cell_size: int = 30,
                 info_width: int = 200):
        """
        get the renderer ready to go
        
        args:
            width: how many cols in game board
            height: how many rows in game board
            cell_size: how big each block is in pixels
            info_width: how wide the side panel is in pixels
        """
        pygame.init()
        
        self.width: int = width
        self.height: int = height
        self.cell_size: int = cell_size
        self.info_width: int = info_width
        
        # figure out window size
        self.game_width: int = width * cell_size
        self.game_height: int = height * cell_size
        self.window_width: int = self.game_width + info_width
        self.window_height: int = self.game_height
        
        # make the window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Tetris AI")
        
        # setup text stuff
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # stuff for animations
        self.clearing_lines: set[int] = set()
        self.clear_start_time = 0
        self.clear_animation_duration = 0.4  # secs
        self.prev_lines_cleared = 0

    def render(self, 
               board_state: np.ndarray[tuple[int, ...], np.dtype[np.int8]],
               score: int,
               lines_cleared: int,
               next_piece: str | None = None) -> None:
        """
        draw the current game state
        
        args:
            board_state: 2d array showing game board
            score: current points
            lines_cleared: how many lines cleared
            next_piece: what piece is coming up (optional)
        """
        # see if we cleared any new lines
        if lines_cleared > self.prev_lines_cleared:
            self.clearing_lines = set()
            # find which lines got cleared
            for y in range(self.height):
                if np.all(board_state[y] != 0):
                    self.clearing_lines.add(y)
            self.clear_start_time = time.time()
        
        # make bg look cool w gradient
        self._draw_background()
        
        # draw game board w fancy effects
        self._draw_board(board_state)
        
        # draw side panel w info
        self._draw_info_panel(score, lines_cleared, next_piece)
        
        # show everything
        pygame.display.flip()
        
        # remember how many lines were cleared
        self.prev_lines_cleared = lines_cleared

    def _draw_background(self):
        """make a gradient bg"""
        for y in range(self.window_height):
            progress = y / self.window_height
            color = self._blend_colors(
                (15, 15, 25),  # dark blue at top
                (35, 35, 45),  # lighter blue at bottom
                progress
            )
            pygame.draw.line(self.screen, color, (0, y), (self.window_width, y))

    def _draw_board(self, board_state: np.ndarray[tuple[int, ...], np.dtype[np.int8]]) -> None:
        """
        draw the game board w cool effects
        
        args:
            board_state: 2d array showing game board
        """
        current_time = time.time()
        animation_progress = (current_time - self.clear_start_time) / self.clear_animation_duration
        
        # draw grid first for bg
        self._draw_grid()
        
        for y in range(self.height):
            for x in range(self.width):
                cell_value = board_state[y][x]
                if cell_value == 0:
                    continue
                
                base_color = self.COLORS.get(cell_value, self.COLORS[0])
                shade_color = self.SHADE_COLORS.get(cell_value, self.COLORS[0])
                highlight_color = self.HIGHLIGHT_COLORS.get(cell_value, self.COLORS[0])
                
                # figure out where block goes
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # add sparkly effects when clearing lines
                if y in self.clearing_lines and animation_progress <= 1.0:
                    # make it fade out
                    fade = 1.0 - animation_progress
                    base_color = self._blend_colors((255, 255, 255), base_color, animation_progress)
                    shade_color = self._blend_colors((255, 255, 255), shade_color, animation_progress)
                    highlight_color = self._blend_colors((255, 255, 255), highlight_color, animation_progress)
                    
                    # add glowy effect
                    glow_rect = rect.inflate(8, 8)
                    glow_color = (255, 255, 255, int(255 * fade))
                    glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=4)
                    self.screen.blit(glow_surf, glow_rect)
                
                # draw block w 3d effect
                self._draw_block(rect, base_color, shade_color, highlight_color)
        
        # cleanup animation stuff when done
        if self.clearing_lines and animation_progress > 1.0:
            self.clearing_lines.clear()

    def _draw_grid(self):
        """draw the bg grid lines"""
        for x in range(self.width + 1):
            pygame.draw.line(
                self.screen,
                (45, 45, 55),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.game_height),
                1
            )
        for y in range(self.height + 1):
            pygame.draw.line(
                self.screen,
                (45, 45, 55),
                (0, y * self.cell_size),
                (self.game_width, y * self.cell_size),
                1
            )

    def _draw_block(self, rect: pygame.Rect, color: tuple[int, int, int],
                   shade_color: tuple[int, int, int],
                   highlight_color: tuple[int, int, int]):
        """draw one block w 3d effect"""
        # main block
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # bottom/right shadow
        shadow_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, shade_color, shadow_rect, border_radius=3)
        
        # top/left highlight
        highlight_rect = rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=2)

    def _blend_colors(self, color1: tuple[int, int, int], 
                     color2: tuple[int, int, int], 
                     factor: float) -> tuple[int, int, int]:
        """
        mix two colors together
        
        args:
            color1: first color (r,g,b)
            color2: second color (r,g,b)
            factor: how much to blend (0.0 to 1.0), where 1.0 is all color2
            
        returns:
            mixed color (r,g,b)
        """
        r = int(color1[0] * (1 - factor) + color2[0] * factor)
        g = int(color1[1] * (1 - factor) + color2[1] * factor)
        b = int(color1[2] * (1 - factor) + color2[2] * factor)
        return (r, g, b)

    def _draw_info_panel(self, 
                        score: int,
                        lines_cleared: int,
                        next_piece: str | None) -> None:
        """
        draw the side panel w game info
        
        args:
            score: current points
            lines_cleared: how many lines cleared
            next_piece: what piece is coming up
        """
        # panel starts after game board
        x_start = self.game_width + 20
        y_start = 20
        
        # show score w shadow effect
        self._draw_text("Score:", (x_start, y_start), shadow=True)
        self._draw_text(str(score), (x_start, y_start + 40), shadow=True)
        
        # show lines cleared
        self._draw_text("Lines:", (x_start, y_start + 100), shadow=True)
        self._draw_text(str(lines_cleared), (x_start, y_start + 140), shadow=True)
        
        # show next piece preview
        if next_piece:
            self._draw_text("Next:", (x_start, y_start + 200), shadow=True)
            
            # make preview box look cool
            preview_rect = pygame.Rect(
                x_start,
                y_start + 240,
                self.cell_size * 4,
                self.cell_size * 4
            )
            pygame.draw.rect(self.screen, (35, 35, 45), preview_rect, border_radius=8)
            pygame.draw.rect(self.screen, (45, 45, 55), preview_rect, border_radius=8, width=1)
            
            # draw next piece in preview
            if next_piece in self.PREVIEW_SHAPES:
                shape = self.PREVIEW_SHAPES[next_piece]
                color = self.COLORS[ord(next_piece)]
                shade_color = self.SHADE_COLORS[ord(next_piece)]
                highlight_color = self.HIGHLIGHT_COLORS[ord(next_piece)]
                
                # center it nicely
                shape_height = len(shape)
                shape_width = len(shape[0])
                offset_x = (4 - shape_width) * self.cell_size // 2
                offset_y = (4 - shape_height) * self.cell_size // 2
                
                # draw the piece
                for y, row in enumerate(shape):
                    for x, cell in enumerate(row):
                        if cell:
                            rect = pygame.Rect(
                                x_start + offset_x + x * self.cell_size,
                                y_start + 240 + offset_y + y * self.cell_size,
                                self.cell_size,
                                self.cell_size
                            )
                            self._draw_block(rect, color, shade_color, highlight_color)

    def _draw_text(self, text: str, pos: tuple[int, int], shadow: bool = False):
        """draw text w optional shadow for style"""
        if shadow:
            shadow_surf = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surf, pos)

    def close(self) -> None:
        """cleanup pygame stuff"""
        pygame.quit() 