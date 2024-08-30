import numpy as np


class Agent:
    def __init__(self, position, maze):
        self.position = np.array(position)
        self.maze = maze

    def move(self, direction):
        self.position += direction
