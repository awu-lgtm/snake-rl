import pygame
import random
from enum import Enum, IntEnum
from collections import namedtuple, deque
import numpy as np
import gymnasium
from gymnasium import spaces

import sys

assets_path = "./assets"
np.set_printoptions(threshold=sys.maxsize)

pygame.init()
font = pygame.font.Font(f'{assets_path}/arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

W = 10
H = 10

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (81, 128, 243)
BLACK = (0,0,0)

SHADOW_GREEN = (148, 189, 70)
DARK_GREEN = (87, 138, 52)
GREEN = (162, 209, 73)
LIGHT_GREEN = (170, 215, 81)

BLOCK_SIZE = 40
SPEED = 2
BORDER_WIDTH = 1/2 * BLOCK_SIZE

# images
apple = pygame.transform.scale(pygame.image.load(f'{assets_path}/apple.png'), (BLOCK_SIZE, 1.27 * BLOCK_SIZE))
snake_head_r = pygame.transform.scale(pygame.image.load(f'{assets_path}/snake_head_h.png'), (1.49 * BLOCK_SIZE, BLOCK_SIZE))
snake_head_l = pygame.transform.flip(snake_head_r, True, False)
snake_head_d = pygame.transform.scale(pygame.image.load(f'{assets_path}/snake_head_v.png'), (BLOCK_SIZE, 1.25 * BLOCK_SIZE))
snake_head_u = pygame.transform.flip(snake_head_d, False, True)

Point = namedtuple('Point', ['x', 'y'])
class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @classmethod
    def is_opposite(self, d1, d2):
        return (d1 == Direction.RIGHT and d2 == Direction.LEFT) \
                or (d1 == Direction.LEFT and d2 == Direction.RIGHT) \
                or (d1 == Direction.UP and d2 == Direction.DOWN) \
                or (d1 == Direction.DOWN and d2 == Direction.UP)
    @classmethod
    def turn_right(self, d):
        return (d + 1)%4

    @classmethod
    def turn_left(self, d):
        return (d + 3)%4 
        
class Square(IntEnum):
    EMPTY = 0
    HEAD = 1
    SNAKE = 2
    FOOD = 3

class Snake:
    def __init__(self, length: int, head: Point, dir: Direction, w: int, h: int):
        self.snake: deque[Point] = deque(maxlen=w*h)
        self.dir: Direction = dir
        self.head: Point = head
        self.snake.append(self.head)
        for i in range(1, length):
            self.snake.append(Point(self.head.x - i, self.head.y))
        self.tail: Point = self.snake[-1]

    def _action_to_move(self, action, x, y):
        match action:
            case Direction.RIGHT:
                x += 1
            case Direction.LEFT:
                x -= 1
            case Direction.UP:
                y -= 1
            case Direction.DOWN:
                y += 1
        return x, y

    def move(self, action: Direction, state: 'State'):
        x,y = self.head.x, self.head.y
        if Direction.is_opposite(self.dir, action):
            x,y = self._action_to_move(self.dir, x, y)
        else:
            x,y = self._action_to_move(action, x, y)
            self.dir = action

        point = Point(x, y)
        if state.is_collision(point): 
            return True, None

        state.state[self.head.y, self.head.x] = Square.SNAKE
        self.head = point
        self.snake.appendleft(self.head)

        if state.is_food(point):
            food = state.snake_eats_food(point)
            state.state[self.head.y, self.head.x] = Square.HEAD
            state.spawn_food()
            return False, food
        
        state.state[self.tail.y, self.tail.x] = Square.EMPTY
        state.state[self.head.y, self.head.x] = Square.HEAD
        self.snake.pop()
        self.tail = self.snake[-1]
        return False, None
    
    def add_to_state(self, state: np.ndarray):
        for i, point in enumerate(self.snake):
            if i == 0:
                state[point.y, point.x] = Square.HEAD
            else:
                state[point.y, point.x] = Square.SNAKE

    def draw_snake(self, display, state: 'State'):
        rgb = (78, 124, 246)
        SHADOW_SIZE = 1/4 * BLOCK_SIZE
        for i, pt in enumerate(self.snake):
            if i == 0:
                match self.dir:
                    case Direction.RIGHT:
                        display.blit(snake_head_r, (pt.x * BLOCK_SIZE - 0.47 * BLOCK_SIZE, pt.y * BLOCK_SIZE))
                        pygame.draw.rect(display, SHADOW_GREEN, pygame.Rect(pt.x * BLOCK_SIZE  - 0.47 * BLOCK_SIZE, pt.y * BLOCK_SIZE + BLOCK_SIZE, 1.49 * BLOCK_SIZE, SHADOW_SIZE))
                    case Direction.LEFT:
                        display.blit(snake_head_l, (pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE))
                        pygame.draw.rect(display, SHADOW_GREEN, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE + BLOCK_SIZE, 1.49 * BLOCK_SIZE, SHADOW_SIZE))
                    case Direction.UP:
                        display.blit(snake_head_u, (pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE))
                    case Direction.DOWN:
                        display.blit(snake_head_d, (pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE - 0.25 * BLOCK_SIZE))
                        pygame.draw.rect(display, SHADOW_GREEN, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE + BLOCK_SIZE, BLOCK_SIZE, SHADOW_SIZE))
            # if i % 2 != 0:
                # State.draw_square(display, BLUE1, pt)
                # pygame.draw.rect(display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + BLOCK_SIZE//10, pt.y * BLOCK_SIZE + BLOCK_SIZE//10, 3 * BLOCK_SIZE//8, 3 * BLOCK_SIZE//8))
            else:
                rgb = (max(rgb[0] - 2 * 0.96**i, 26), max(rgb[1] - 2* 0.96**i, 70), max(rgb[2] - 3 * 0.96**i, 163))
                if state.is_empty(Point(pt.x, pt.y + 1)) or state.is_food(Point(pt.x, pt.y + 1)):
                    pygame.draw.rect(display, SHADOW_GREEN, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE + BLOCK_SIZE, BLOCK_SIZE, SHADOW_SIZE))
                State.draw_square(display, (int(rgb[0]), int(rgb[1]), int(rgb[2])), pt)


class Food:
    def __init__(self, value: int, point: Point):
        self.value: int = value
        self.point: Point = point

        
class State:
    def __init__(self, w, h):
        self.score = 0
        self.w, self.h = w, h
        self._init_state()

    def _init_state(self):
        w,h = self.w, self.h
        self.state: np.ndarray = np.zeros((w, h))
        self.snake: Snake = Snake(3, Point(2, h//2), Direction.RIGHT, w, h)
        self.snake.add_to_state(self.state)
        self.foods: list[Food] = []

    def reset(self):
        self.score = 0
        self._init_state()
    
    def spawn_food(self, count=1):
        empties = np.argwhere(self.state == Square.EMPTY)
        count = min(count, len(empties))
        rng = np.random.default_rng()
        points = rng.choice(empties, count, replace=False)
        for point in points:
            self.state[point[0], point[1]] = Square.FOOD
            point = Point(point[1], point[0])
            food = Food(1, point)
            self.foods.append(food)
    
    def move_snake(self, action: Direction):
        done, food = self.snake.move(action, self)
        reward = food.value if food is not None else 0
        if self.is_win():
            self.score += 100
            reward += 100
            return True, reward
        return done, reward
    
    def is_win(self):
        return len(self.snake.snake) == self.w * self.h
    
    def is_out_of_bounds(self, point: Point):
        x,y = point.x, point.y
        w,h = self.state.shape
        return x >= w or x < 0 or y >= h or y < 0
    
    def is_collision(self, point: Point):
        x,y = point.x, point.y
        return self.is_out_of_bounds(point) or self.state[y, x] == Square.SNAKE or self.state[y, x] == Square.HEAD
    
    def is_food(self, point: Point):
        x,y = point.x, point.y
        return not self.is_out_of_bounds(point) and self.state[y, x] == Square.FOOD
    
    def is_empty(self, point: Point):
        x,y = point.x, point.y
        return not self.is_out_of_bounds(point) and self.state[y, x] == Square.EMPTY
    
    def win(self):
        self.score += 100

    def snake_eats_food(self, point: Point):
        x,y = point.x, point.y
        for i, food in enumerate(self.foods):
            if food.point == point:
                self.state[y, x] = Square.EMPTY
                self.score += food.value
                break
        del self.foods[i]
        return food

    def draw_board(self, display):
        for y in range(len(self.state)):
            for x in range(len(self.state[y])):
                point = Point(x, y)
                if (x + y)%2 == 0:
                    self.draw_square(display, LIGHT_GREEN, point)
                else:
                    self.draw_square(display, GREEN, point)
    
    def draw_food(self, display):
        for food in self.foods:
            display.blit(apple, (food.point.x * BLOCK_SIZE, food.point.y * BLOCK_SIZE - 0.135 * BLOCK_SIZE))

    @classmethod
    def draw_square(self, display, color, point):
        x,y = point.x, point.y
        pygame.draw.rect(display, color, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def draw_state(self, display):
        self.draw_board(display)
        self.snake.draw_snake(display, self)
        self.draw_food(display)

    def get_state_copy(self):
        return self.state.copy()
    
class SnakeGame:
    display: pygame.display
    def __init__(self, food_count, w=W, h=H):
        self.w = W
        self.h = H

        self.state = State(w, h)
        self.state.spawn_food(food_count)
    
    def draw_border(self):
        for x in range(self.w + 1):
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(x * BLOCK_SIZE, 0, BLOCK_SIZE, 1/2 * BLOCK_SIZE))
        
        for x in range(self.w + 1):
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(x * BLOCK_SIZE, self.h * BLOCK_SIZE + BORDER_WIDTH, BLOCK_SIZE, 1/2 * BLOCK_SIZE))
        
        for y in range(self.h + 1):
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(0, y * BLOCK_SIZE, 1/2 * BLOCK_SIZE, BLOCK_SIZE))
        
        for y in range(self.h + 1):
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(self.w * BLOCK_SIZE + BORDER_WIDTH, y * BLOCK_SIZE, 1/2 * BLOCK_SIZE, BLOCK_SIZE))
    
    def _render_frame(self):
        self.draw_border()
        self.state.draw_state(self.board_display)
        
        text = font.render("Score: " + str(self.state.score), True, WHITE)
        self.display.blit(self.board_display, (BORDER_WIDTH, BORDER_WIDTH))
        self.display.blit(text, [0, 0])

class SnakeEnv(SnakeGame, gym.Env):
    metadata = {'render.modes': ['human'], "render_fps": 8}
    def __init__(self, render_mode=None, w=W, h=H):
        Snake.__init__(self, 10, w, h)
        
        self.observation_space = spaces.MultiDiscrete(np.empty(w*h).fill(3))
        self.action_space = spaces.Discrete(3)
        self.game_state = State(w, h)
        self.game_state.spawn_food(5)

        self.display = None
        self.board_display = None
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def _action_to_direction(self, action):
        dir = self.game_state.snake.dir
        match action:
            case 0:
                return dir
            case 1:
                return Direction.turn_left(dir)
            case 2:
                return Direction.turn_left(dir)
    
    def step(self, action):
        dir = self._action_to_direction(action)
        done, reward = self.game_state.move_snake(dir)
        next_obs = self.game_state.get_state_copy().flatten()
        info = dict()

        if self.render_mode == "human":
            self._render_frame()

        return next_obs, done, reward, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game_state.reset()
        obs = self.game_state.get_state_copy().flatten()

        if self.render_mode == "human":
            self._render_frame()

        return obs
    
    def render(self):
        self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.w, self.h))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        super()._render_frame()

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class SnakePygame(SnakeGame):
    def __init__(self, w=640, h=480):
        SnakeGame.__init__(self, 10, w, h)
        # init display
        self.display = pygame.display.set_mode((self.w * BLOCK_SIZE + 2 * BORDER_WIDTH, self.h * BLOCK_SIZE + 2 * BORDER_WIDTH))
        self.board_display = pygame.surface.Surface((self.w * BLOCK_SIZE, self.h * BLOCK_SIZE))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

    def play_step(self):
        dir = self.state.snake.dir
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    dir = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    dir = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    dir = Direction.UP
                elif event.key == pygame.K_DOWN:
                    dir = Direction.DOWN
        
        # 2. move
        done, _ = self.state.move_snake(dir)
        
        # 3. check if done
        if done:
            return self.state.state, done, self.state.score
        
        # 4. update ui and clock
        self._render_frame()
        self.clock.tick(SPEED)
        # 5. return game over and score
        return self.state.state, done, self.state.score
    
    def _render_frame(self):
        super()._render_frame()
        pygame.display.flip()
            

if __name__ == '__main__':
    game = SnakePygame(W, H)
        
            # game loop
    while True:
        state, done, score = game.play_step()
        print(state)
        
        if done:
            break

    print('Final Score', score)
        
        
    pygame.quit()