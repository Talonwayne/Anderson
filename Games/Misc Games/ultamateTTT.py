import pygame
import sys
from pygame.locals import *
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
LINE_WIDTH = 5
BOARD_ROWS = 9
BOARD_COLS = 9
SQUARE_SIZE = SCREEN_WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
LARGE_SYMBOL_WIDTH = 50
LARGE_SYMBOL_SPACE = 75

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (84, 84, 84)
DRAW_COLOR = (0, 0, 255)  # Blue box for draw
HIGHLIGHT_COLOR = (255, 255, 0)

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Ultimate Tic Tac Toe')
screen.fill(BG_COLOR)

# Board
board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
main_board = [[" " for _ in range(3)] for _ in range(3)]

# Current player
current_player = "X"
target_board = None

# Scoreboard
scoreboard = {"X": 0, "O": 0}

# Game font for "D"
font = pygame.font.Font(None, 144)


def draw_lines():
    # Horizontal and Vertical lines
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, LINE_COLOR, (0, row * SQUARE_SIZE), (SCREEN_WIDTH, row * SQUARE_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, LINE_COLOR, (col * SQUARE_SIZE, 0), (col * SQUARE_SIZE, SCREEN_HEIGHT), LINE_WIDTH)

    # Main board horizontal and vertical lines
    for row in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (0, row * 3 * SQUARE_SIZE), (SCREEN_WIDTH, row * 3 * SQUARE_SIZE),
                         LINE_WIDTH * 2)
    for col in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (col * 3 * SQUARE_SIZE, 0), (col * 3 * SQUARE_SIZE, SCREEN_HEIGHT),
                         LINE_WIDTH * 2)


def draw_figures():
    for main_row in range(3):
        for main_col in range(3):
            if main_board[main_row][main_col] in ['X', 'O', 'D']:
                draw_large_symbol(main_row, main_col, main_board[main_row][main_col])
            else:
                for sub_row in range(3):
                    for sub_col in range(3):
                        if board[main_row][main_col][sub_row][sub_col] == 'X':
                            draw_cross(main_row, main_col, sub_row, sub_col)
                        elif board[main_row][main_col][sub_row][sub_col] == 'O':
                            draw_circle(main_row, main_col, sub_row, sub_col)


def draw_circle(main_row, main_col, sub_row, sub_col):
    pygame.draw.circle(screen, CIRCLE_COLOR, (main_col * 3 * SQUARE_SIZE + sub_col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                              main_row * 3 * SQUARE_SIZE + sub_row * SQUARE_SIZE + SQUARE_SIZE // 2),
                       CIRCLE_RADIUS, CIRCLE_WIDTH)


def draw_cross(main_row, main_col, sub_row, sub_col):
    pygame.draw.line(screen, CROSS_COLOR, (main_col * 3 * SQUARE_SIZE + sub_col * SQUARE_SIZE + SPACE,
                                           main_row * 3 * SQUARE_SIZE + sub_row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                     (main_col * 3 * SQUARE_SIZE + sub_col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                      main_row * 3 * SQUARE_SIZE + sub_row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
    pygame.draw.line(screen, CROSS_COLOR, (main_col * 3 * SQUARE_SIZE + sub_col * SQUARE_SIZE + SPACE,
                                           main_row * 3 * SQUARE_SIZE + sub_row * SQUARE_SIZE + SPACE),
                     (main_col * 3 * SQUARE_SIZE + sub_col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                      main_row * 3 * SQUARE_SIZE + sub_row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)


def draw_large_symbol(main_row, main_col, symbol):
    center_x = main_col * 3 * SQUARE_SIZE + 1.5 * SQUARE_SIZE
    center_y = main_row * 3 * SQUARE_SIZE + 1.5 * SQUARE_SIZE
    if symbol == 'X':
        pygame.draw.line(screen, CROSS_COLOR, (center_x - LARGE_SYMBOL_SPACE, center_y - LARGE_SYMBOL_SPACE),
                         (center_x + LARGE_SYMBOL_SPACE, center_y + LARGE_SYMBOL_SPACE), LARGE_SYMBOL_WIDTH)
        pygame.draw.line(screen, CROSS_COLOR, (center_x - LARGE_SYMBOL_SPACE, center_y + LARGE_SYMBOL_SPACE),
                         (center_x + LARGE_SYMBOL_SPACE, center_y - LARGE_SYMBOL_SPACE), LARGE_SYMBOL_WIDTH)
    elif symbol == 'O':
        pygame.draw.circle(screen, CIRCLE_COLOR, (center_x, center_y), SQUARE_SIZE, LARGE_SYMBOL_WIDTH)
    elif symbol == 'D':
        pygame.draw.rect(screen, DRAW_COLOR,
                         (main_col * 3 * SQUARE_SIZE, main_row * 3 * SQUARE_SIZE, 3 * SQUARE_SIZE, 3 * SQUARE_SIZE))


def draw_highlights():
    if target_board:
        main_row, main_col = target_board
        pygame.draw.rect(screen, HIGHLIGHT_COLOR,
                         (main_col * 3 * SQUARE_SIZE, main_row * 3 * SQUARE_SIZE, 3 * SQUARE_SIZE, 3 * SQUARE_SIZE),
                         LINE_WIDTH)
    else:
        for main_row in range(3):
            for main_col in range(3):
                if main_board[main_row][main_col] == " ":
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (
                    main_col * 3 * SQUARE_SIZE, main_row * 3 * SQUARE_SIZE, 3 * SQUARE_SIZE, 3 * SQUARE_SIZE),
                                     LINE_WIDTH)


def check_winner(board, player):
    for row in board:
        if all([spot == player for spot in row]):
            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False


def is_board_full(board):
    return all([spot is not None for row in board for spot in row])


def reset_game():
    global board, main_board, current_player, target_board
    board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
    main_board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
    target_board = None
    screen.fill(BG_COLOR)
    draw_lines()
    draw_scoreboard()


def draw_scoreboard():
    font = pygame.font.Font(None, 74)
    x_score = font.render(f'X: {scoreboard["X"]}', True, CROSS_COLOR)
    o_score = font.render(f'O: {scoreboard["O"]}', True, CIRCLE_COLOR)
    screen.blit(x_score, (20, 20))
    screen.blit(o_score, (SCREEN_WIDTH - 150, 20))


def handle_click(row, col):
    global current_player, target_board
    main_row, sub_row = divmod(row, 3)
    main_col, sub_col = divmod(col, 3)
    if target_board and (main_row != target_board[0] or main_col != target_board[1]):
        return
    if board[main_row][main_col][sub_row][sub_col] is None:
        board[main_row][main_col][sub_row][sub_col] = current_player
        if check_winner(board[main_row][main_col], current_player):
            main_board[main_row][main_col] = current_player
        if is_board_full(board[main_row][main_col]) and main_board[main_row][main_col] == " ":
            main_board[main_row][main_col] = "D"
        if check_winner(main_board, current_player):
            draw_figures()
            pygame.display.update()
            time.sleep(1)
            scoreboard[current_player] += 1
            reset_game()
            return
        if all([all([cell in ['X', 'O', 'D'] for cell in row]) for row in main_board]):
            print("The game is a draw!")
            reset_game()
            return
        current_player = "O" if current_player == "X" else "X"
        target_board = (sub_row, sub_col) if main_board[sub_row][sub_col] == " " else None
        screen.fill(BG_COLOR)
        draw_lines()
        draw_figures()
        draw_highlights()
        draw_scoreboard()


# Main game loop
reset_game()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            mouseX = event.pos[0]
            mouseY = event.pos[1]
            clicked_row = mouseY // SQUARE_SIZE
            clicked_col = mouseX // SQUARE_SIZE
            handle_click(clicked_row, clicked_col)

    pygame.display.update()
