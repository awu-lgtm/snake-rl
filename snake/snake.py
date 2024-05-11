import pygame
from enum import IntEnum
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from utils import Point, Direction, l1_norm, dirs_to_point, map_dirs, move_in_dir
import sys
from pathlib import Path

file_path = Path(__file__).resolve().parent

assets_path = f"{file_path}/assets"
np.set_printoptions(threshold=sys.maxsize)

pygame.init()
font = pygame.font.Font(f'{assets_path}/arial.ttf', 25)

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
BORDER_WIDTH = 1/2 * BLOCK_SIZE

# images
apple = pygame.transform.scale(pygame.image.load(f'{assets_path}/apple.png'), (BLOCK_SIZE, 1.27 * BLOCK_SIZE))
snake_head_r = pygame.transform.scale(pygame.image.load(f'{assets_path}/snake_head_h.png'), (1.49 * BLOCK_SIZE, BLOCK_SIZE))
snake_head_l = pygame.transform.flip(snake_head_r, True, False)
snake_head_d = pygame.transform.scale(pygame.image.load(f'{assets_path}/snake_head_v.png'), (BLOCK_SIZE, 1.25 * BLOCK_SIZE))
snake_head_u = pygame.transform.flip(snake_head_d, False, True)
        
class Square(IntEnum):
    EMPTY = 0
    FOOD = 2
    HEAD_UP = 3
    HEAD_RIGHT = 4
    HEAD_DOWN = 5
    HEAD_LEFT = 6
    SNAKE = 7

    @classmethod
    def is_snake(self, square):
        return square >= Square.HEAD_UP
    
    @classmethod
    def head_dir_to_square(self, dir):
        return dir + Square.HEAD_UP


class Snake:
    def __init__(self, length: int, head: Point, dir: Direction, w: int, h: int):
        self.snake: deque[Point] = deque(maxlen=w*h)
        self.dir: Direction = dir
        self.head: Point = head
        self.snake.append(self.head)
        for i in range(1, length):
            self.snake.append(Point(self.head.x - i, self.head.y))
        self.tail: Point = self.snake[-1]

    def move(self, action: Direction, state: 'State'):
        if Direction.is_opposite(self.dir, action):
            point = move_in_dir(self.head, self.dir)
        else:
            point = move_in_dir(self.head, action)
            self.dir = action

        if state.is_out_of_bounds(point):
            return True, None, (True, False)
        if state.is_snake(point):
            return True, None, (False, True)

        state.set_state(self.head, Square.SNAKE)
        self.head = point
        self.snake.appendleft(self.head)

        if state.is_food(point):
            food = state.snake_eats_food(point)
            state.set_state(self.head, Square.head_dir_to_square(self.dir))
            if state.fully_obs_state is not None:
                self.add_to_state(state)
            state.spawn_food()
            return False, food, (False, False)
        
        state.set_state(self.tail, Square.EMPTY)
        state.set_state(self.head, Square.head_dir_to_square(self.dir))
        self.snake.pop()
        self.tail = self.snake[-1]
        if state.fully_obs_state is not None:
            self.add_to_state(state)
        return False, None, (False, False)
    
    def add_to_state(self, state: 'State'):
        for i, point in enumerate(self.snake):
            if i == 0:
                Square.head_dir_to_square(self.dir)
                state.set_state(point, Square.head_dir_to_square(self.dir))
            else:
                state.set_state(point, Square.SNAKE, Square.SNAKE + i - 1)

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
    def __init__(self, w, h, food_count, win_reward=1, loss_reward=-1, eat_reward=0.1, fully_obs_state=True):
        self.score = 0
        self.w, self.h = w, h
        self.food_count = food_count
        self.win_reward = win_reward
        self.eat_reward = eat_reward
        self.loss_reward = loss_reward

        self._init_state(fully_obs_state)

    def _init_state(self, use_fully_obs_state):
        w,h = self.w, self.h
        self.state: np.ndarray = np.zeros((h, w), dtype=np.int8)
        self.fully_obs_state = self.state.copy() if use_fully_obs_state else None
        self.snake: Snake = Snake(3, Point(2, h//2), Direction.RIGHT, w, h)
        self.foods: list[Food] = []
        self.snake.add_to_state(self)
        self.spawn_food(self.food_count)

    def set_state(self, point: Point, val: int, fully_obs_val = None):
        x,y = point.x, point.y
        self.state[y, x] = val
        if self.fully_obs_state is not None: 
            self.fully_obs_state[y, x] = fully_obs_val if fully_obs_val else val
    
    def reset(self):
        self.score = 0
        use_fully_obs_state = self.fully_obs_state is not None
        self._init_state(use_fully_obs_state)
    
    def spawn_food(self, count=1):
        empties = np.argwhere(self.state == Square.EMPTY)
        count = min(count, len(empties))
        rng = np.random.default_rng()
        points = rng.choice(empties, count, replace=False)
        for point in points:
            point = Point(point[1], point[0])
            self.set_state(point, Square.FOOD)
            food = Food(self.eat_reward, point)
            self.foods.append(food)
    
    def move_snake(self, action: Direction):
        loss, food, collision_conds = self.snake.move(action, self)
        reward = food.value if food is not None else 0
        if self.is_win():
            self.score += self.win_reward
            reward += self.win_reward
            return True, reward, collision_conds
        if loss:
            reward += self.loss_reward
        return loss, reward, collision_conds
    
    def is_win(self):
        return len(self.snake.snake) == self.w * self.h
    
    def is_out_of_bounds(self, point: Point):
        x,y = point.x, point.y
        h,w = self.state.shape
        return x >= w or x < 0 or y >= h or y < 0
    
    def is_snake(self, point: Point):
        x,y = point.x, point.y
        return Square.is_snake(self.state[y, x])
    
    def is_collision(self, point: Point):
        return self.is_out_of_bounds(point) or self.is_snake(point)
    
    def is_food(self, point: Point):
        x,y = point.x, point.y
        return not self.is_out_of_bounds(point) and self.state[y, x] == Square.FOOD
    
    def is_empty(self, point: Point):
        x,y = point.x, point.y
        return not self.is_out_of_bounds(point) and self.state[y, x] == Square.EMPTY

    def snake_eats_food(self, point: Point):
        for i, food in enumerate(self.foods):
            if food.point == point:
                self.set_state(point, Square.EMPTY)
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

    def get_state_copy(self, as_image=False):
        if as_image:
            return np.reshape(self.fully_obs_state.copy(), (self.h, self.w, 1))
        return self.fully_obs_state.copy()
    
class SnakeGame:
    display: pygame.display
    def __init__(self, w=W, h=H, food_count=1):
        self.w = w
        self.h = h

        self.state = State(w, h, food_count)
    
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
    metadata = {'render.modes': ['human'], "render_fps": 16}
    def __init__(self, render_mode=None, w=W, h=H, food_count=1,
                 head_relative_action=True, head_relative_state=True, 
                 absolute_state=True, as_image=False, normalize=False, 
                 truncation_lim=None, truncation_reward=-1):
        SnakeGame.__init__(self, w=w, h=h, food_count=food_count)
        
        self.absolute_state = absolute_state
        self.head_relative_state = head_relative_state
        self.as_image = as_image
        self.normalize = normalize

        observation_space = {}
        if absolute_state:
            if as_image:
                observation_space['state'] = Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
            else:
                high = w * h + Square.SNAKE - 2 if not normalize else 1
                observation_space['state'] = Box(low=0, high=high, shape=(h, w), dtype=np.uint8)
        if head_relative_state: 
            observation_space['head-relative'] = Dict({
                'dirs-to-food': MultiBinary(4), 
                'danger-dirs': MultiBinary(3),
                'head-dir': MultiBinary(4)
            })
        self.observation_space = Dict(observation_space)
        self.action_space = Discrete(3) if head_relative_action else Discrete(4)

        self.render_mode = render_mode
        self.i = 0
        self.truncation_lim = truncation_lim if truncation_lim is not None else (w*h)**2
        self.truncation_reward = truncation_reward

        self.display = None
        self.board_display = None
        self.clock = None
    
    def _action_to_direction(self, action):
        dir = self.state.snake.dir
        match action:
            case 0:
                return dir
            case 1:
                return Direction.turn_left(dir)
            case 2:
                return Direction.turn_right(dir)
    
    def step(self, action):
        self.i += 1
        truncated = self.i >= self.truncation_lim
        if truncated:
            self.state.score += self.truncation_reward

        dir = action
        if self.action_space.n == 3:
            dir = self._action_to_direction(action)
        terminated, reward, collision_conds = self.state.move_snake(dir)
        next_obs = self._get_obs()
        info = {'dir': dir, "collision_conds": collision_conds, "state": self.state.state, "score": self.state.score}

        if self.render_mode == "human":
            self._render_frame()
        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state.reset()
        self.i = 0
        obs = self._get_obs()
        info = {'dir': Direction.RIGHT, "collision_conds": (False, False), "state": self.state, "score": self.state.score}

        if self.render_mode == "human":
            self._render_frame()
        return obs, info
    
    def _get_obs(self):
        obs = {}
        if self.absolute_state:
            if self.as_image:
                state = self.state.get_state_copy(as_image=True)
            else:
                state = self.state.get_state_copy().flatten()
            obs['state'] = state
        if self.head_relative_state:
            head = self.state.snake.head
            if len(self.state.foods) == 0:
                dirs_to_food = np.zeros(4)
            else:
                nearest_food = min(self.state.foods, key=lambda food: l1_norm(food.point, head))
                dirs_to_food = dirs_to_point(head, nearest_food.point)
            dangers = map_dirs(head, lambda p: int(self.state.is_collision(p)), dir=self.state.snake.dir)
            head_dir = np.zeros(4)
            head_dir[self.state.snake.dir] = 1
            obs['head-relative'] = {'dirs-to-food': dirs_to_food, 'danger-dirs': dangers, 'head-dir': head_dir}
        return obs

    def render(self):
        self._render_frame()

    def _render_frame(self):
        if self.display is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.display = pygame.display.set_mode((self.w * BLOCK_SIZE + 2 * BORDER_WIDTH, self.h * BLOCK_SIZE + 2 * BORDER_WIDTH))
            self.board_display = pygame.surface.Surface((self.w * BLOCK_SIZE, self.h * BLOCK_SIZE))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        super()._render_frame()

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.display is not None:
            pygame.display.quit()
            pygame.quit()

class SnakePygame(SnakeGame):
    def __init__(self, w, h, game_speed):
        SnakeGame.__init__(self, w, h, 40)
        # init display
        self.display = pygame.display.set_mode((self.w * BLOCK_SIZE + 2 * BORDER_WIDTH, self.h * BLOCK_SIZE + 2 * BORDER_WIDTH))
        self.board_display = pygame.surface.Surface((self.w * BLOCK_SIZE, self.h * BLOCK_SIZE))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.game_speed = game_speed

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
        done, _, _ = self.state.move_snake(dir)
        
        # 3. check if done
        if done:
            return self.state.fully_obs_state, done, self.state.score
        
        # 4. update ui and clock
        self._render_frame()
        self.clock.tick(self.game_speed)
        # 5. return game over and score
        return self.state.fully_obs_state, done, self.state.score
    
    def _render_frame(self):
        super()._render_frame()
        pygame.display.flip()
            

if __name__ == '__main__':
    game = SnakePygame(W, H, 5)
        
            # game loop
    while True:
        state, done, score = game.play_step()
        print(state)
        if done:
            break

    print('Final Score', score)
        
        
    pygame.quit()