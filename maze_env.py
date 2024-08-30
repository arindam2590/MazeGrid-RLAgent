import numpy as np
import pygame
import heapq


class MazeEnv:
    def __init__(self, params):
        self.size = params['SIZE']
        self.cell_size = params['CELL_SIZE']
        self.maze = None
        self.directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.walls = []
        self.window_size = (self.cell_size * self.size, self.cell_size * self.size)
        self.screen = None
        self.clock = None
        self.WHITE = (255, 255, 255)  # white color

    def env_setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Maze Simulation')
        self.clock = pygame.time.Clock()

    def generate_maze(self):
        start_x, start_y = np.random.choice(range(1, self.size)), 1
        self.maze[start_x, start_y] = 0
        for direction in self.directions:
            nx, ny = start_x + direction[0], start_y + direction[1]
            if 1 < nx < self.size - 1 and 1 < ny < self.size - 1:
                self.walls.append((nx, ny, start_x, start_y))

        while self.walls:
            idx_no = len(self.walls)
            wall = self.walls[np.random.choice(idx_no)]
            x, y, px, py = wall
            if self.maze[x, y] == 1:
                adjacent_passages = sum(
                    self.size - 1 > x + direction[0] >= 1 and
                    self.size - 1 > y + direction[1] >= 1 and
                    self.maze[x + direction[0], y + direction[1]] == 0 for direction in self.directions
                )

                if adjacent_passages == 1:
                    self.maze[x, y] = 0
                    for direction in self.directions:
                        nx, ny = x + direction[0], y + direction[1]
                        if self.size - 1 > nx > 0 and self.size - 1 > ny > 0 and self.maze[nx, ny] == 1:
                            self.walls.append((nx, ny, x, y))
            self.walls.remove(wall)

    def is_valid_position(self, position):
        x, y = position
        return 1 <= x < self.size - 1 and 1 <= y < self.size - 1 and self.maze[y, x] == 0

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start, goal):
        if isinstance(start, np.ndarray):
            start = tuple(start)
        if isinstance(goal, np.ndarray):
            goal = tuple(goal)

        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if self.is_valid_position(neighbor):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                        if neighbor not in [i[2] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

        return None
