from snake.snake import SnakeEnv

w,h = 9,10
food_count = 1

def make(absolute_state=False, head_relative_state=False, as_image=False, render_mode=None):
    return SnakeEnv(w=w, h=h, food_count=food_count, absolute_state=absolute_state, head_relative_state=head_relative_state, as_image=as_image, render_mode=render_mode)

def snake_head_relative():
    return make(head_relative_state=True)

def snake_absolute_state():
    return make(absolute_state=True)

def snake_image():
    return make(as_image=True)

def snake_all():
    return make(as_image=True, head_relative_state=True, absolute_state=True)