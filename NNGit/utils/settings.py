import pygame
import numpy as np
from random import *
import matplotlib.pyplot as pt
from matplotlib.animation import FuncAnimation

pygame.init()
pygame.font.init()

### PYGAME SETTINGS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (30, 30, 30)
GREY = (75, 75, 75)

FPS = 200
WIDTH, HEIGHT = 600, 700

ROWS = COLS = 20 #MAX 165
TOOLBAR_HEIGHT = HEIGHT - WIDTH
PIXEL_SIZE = WIDTH // COLS

BG_COLOUR = WHITE

DRAW_GRID_LINES = True

def  get_font(size):
    return pygame.font.SysFont("comicsans", size)


### NN SETTINGS

ACTIVATION = np.vectorize(lambda x : 1/(1 + np.exp(-x)))
ACTIVATION_DERIVATIVE = np.vectorize(lambda x: ACTIVATION(x) * (1-ACTIVATION(x)))

WEIGHT_RNG = lambda: uniform(-0.5, 0.5)
# currently 4 output nodes

CONVERSION = {"cross": 0, "circle": 1, "triangle":2, "line":3}
SHAPES = dict(map(lambda x: (x[1],x[0]), CONVERSION.items()))
EXPECTED = {0: [1,0,0,0], 1:[0,1,0,0], 2:[0,0,1,0], 3:[0,0,0,1]}
LEARNRATE = 0.3
###

class Picture:
    def __init__(self, grid, classification):
        self.grid = grid
        self.classification = classification
    def __str__(self):
        return self.classification





### MISC
def convert(grid):
    return list(map(lambda y: list(map(lambda x: 1 - sum(x)/(3*255), y)), grid)) 