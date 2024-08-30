import os
import time
import pygame
import numpy as np
from maze_env import MazeEnv
from agent import Agent
from dqn import DQNModel, DoubleDQNModel, DuelingDQNModel, DRQNModel


class Simulation:
    def __init__(self, params, mode, render=False):
        self.train_mode = mode
        self.model = params['MODEL_NAME']
        self.render = render
        self.maze_size = params['SIZE']
        self.state_size = params['SIZE'] * params['SIZE']
        self.action_size = params['ACTION_SIZE']
        self.batch_size = params['BATCH_SIZE']
        self.fps = params['FPS']
        self.maze = None
        self.path = None
        self.agent = None
        self.source = None
        self.destination = None
        self.game_steps = 0
        self.prev_goal_dist = np.inf
        self.model_save_path = params['MAZE_MODEL_DIR']
        self.dqn_model = None
        self.running = True
        self.train_start_time = None
        self.train_end_time = None
        self.is_env_initialized = False

    def game_initialize(self, params):
        maze_dir = params['MAZE_DATA_DIR']
        if not os.path.exists(maze_dir):
            os.makedirs(maze_dir)

        self.maze = MazeEnv(params)
        if self.train_mode:
            self.maze.maze = np.ones((params['SIZE'], params['SIZE']), dtype=int)
            self.maze.generate_maze()
            np.save(maze_dir + '/mazegrid.npy', self.maze.maze)
        else:
            self.maze.maze = np.load(maze_dir + '/mazegrid.npy')

        if self.render:
            self.maze.env_setup()
        self._generate_src_dst(params)
        self.path = self.maze.a_star(self.source, self.destination)
        self.agent = Agent(self.source, self.maze)
        if self.model == 'DQN':
            self.dqn_model = DQNModel(self.state_size, self.action_size, self.maze,
                                      self.train_mode, self.model_save_path, params)
            print(f'\n\nInfo: DQN Model has been selected for the Training and Testing of Agent...')
        elif self.model == 'DoubleDQN':
            self.dqn_model = DoubleDQNModel(self.state_size, self.action_size, self.maze,
                                            self.train_mode, self.model_save_path, params)
            print(f'\n\nInfo: Double DQN Model has been selected for the Training and Testing of Agent...')
        elif self.model == 'DuelingDQN':
            self.dqn_model = DuelingDQNModel(self.state_size, self.action_size, self.maze,
                                             self.train_mode, self.model_save_path, params)
            print(f'\n\nInfo: Dueling DQN Model has been selected for the Training and Testing of Agent...')

        elif self.model == 'DRQN':
            self.dqn_model = DRQNModel(self.state_size, self.action_size, self.maze,
                                       self.train_mode, self.model_save_path, params)
            print(f'\n\nInfo: DRQN Model has been selected for the Training and Testing of Agent...')
        else:
            print(f'\n\nInfo: Please select a valid Model for the Training and Testing of Agent...')

        fig_dir = params['FIG_DIR']
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        self.is_env_initialized = True

    def _generate_src_dst(self, params):
        self.source = np.array((1, np.random.choice(range(1, self.maze_size))))
        self.destination = np.array((self.maze_size - 2, np.random.choice(range(1, self.maze_size))))

        while not self.maze.is_valid_position(self.source):
            self.source = np.array((1, np.random.choice(range(1, self.maze_size))))
        while not self.maze.is_valid_position(self.destination):
            self.destination = np.array((self.maze_size - 2, np.random.choice(range(1, self.maze_size))))

        if self.source is not None and self.destination is not None:
            data_dir = params['MAZE_DATA_DIR']
            location = np.vstack((self.source, self.destination))
            np.save(data_dir+'/source_destination.npy', location)

    def close_window(self):
        pygame.quit() if self.render else None

    def event_on_game_window(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def step(self, action):
        self.game_steps += 1
        terminated, truncated = False, False
        info = {'Success': False}
        new_position = [self.agent.position[0] + self.maze.directions[action][0],
                        self.agent.position[1] + self.maze.directions[action][1]]
        if self.maze.is_valid_position(new_position):
            if np.array_equal(self.agent.position, self.destination):
                reward = 100
                print(f'\nInfo: HURRAY!! Agent has reached its destination...')
                self.maze_cell_visited.append((3, reward))
                info['Success'] = True
                terminated = True
                self.game_steps = 0
            else:
                reward = -1
                self.maze_cell_visited.append((0, reward))
                dist = np.linalg.norm(self.destination - self.agent.position)
                if self.game_steps < 50:
                    if self.prev_goal_dist > dist:
                        reward += 5
                        self.prev_goal_dist = dist
                else:
                    truncated = True
                    self.game_steps = 0
        else:
            reward = -75
            self.maze_cell_visited.append((1, reward))
            terminated = True
            self.game_steps = 0
        direction = np.array((self.maze.directions[action][0], self.maze.directions[action][1]))
        self.agent.move(direction)

        return self._get_state(), reward, terminated, truncated, info

    def reset(self):
        self.agent.position = np.array(self.source)
        self.maze_cell_visited = []
        return self._get_state()

    def _get_state(self):
        one_hot_state = self.maze.maze.flatten()
        agent_index = self.agent.position[0] * self.maze_size + self.agent.position[1]
        one_hot_state[agent_index] = 2
        target_index = self.destination[0] * self.maze_size + self.destination[1]
        one_hot_state[target_index] = 3
        return one_hot_state

    def train_agent(self, episodes):
        print(f'Info: Agent Training has been started over the Maze Simulation...')
        print(f'Info: Source: {self.source} Destination: {self.destination}')
        self.train_start_time = time.time()

        # Training Code
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        rewards_per_episode = np.zeros(episodes)
        epsilon_decay_per_episode = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        training_error = np.zeros(episodes)
        success_rate = np.zeros(episodes)
        for episode in range(episodes):
            state = self.reset()
            done, total_reward, step, success_status, loss = False, 0, 0, 0, 0.0
            while not done:
                self.update_display() if self.render else None
                if step % self.dqn_model.update_rate == 0:
                    self.dqn_model.update_target_network()
                action = self.dqn_model.act(state)
                next_state, reward, terminated, truncated, info = self.step(action)

                self.dqn_model.remember(state, action, reward, next_state, done)

                state = next_state
                step += 1
                done = terminated or truncated
                total_reward += reward

                if len(self.dqn_model.replay_buffer.buffer) > self.batch_size:
                    loss = self.dqn_model.train(self.batch_size)

                if info['Success']:
                    success_status = 1
            self.dqn_model.epsilon = max(self.dqn_model.epsilon - self.dqn_model.epsilon_decay, 0)
            print(f'Episode {episode + 1}/{episodes} - Steps: {step}, Epsilon: '
                  f'{self.dqn_model.epsilon:.3f} Reward: {total_reward:.2f} \nVisited cells: {self.maze_cell_visited}')
            rewards_per_episode[episode] = total_reward
            epsilon_decay_per_episode[episode] = self.dqn_model.epsilon
            steps_per_episode[episode] = step
            success_rate[episode] = success_status
            if self.train_mode:
                training_error[episode] = loss

            if self.dqn_model.epsilon == 0.0:
                self.dqn_model.alpha = 0.0001

        self.dqn_model.main_network.save(
            self.model_save_path + '/' + self.model + '_model_final.keras') if self.render else None
        print("Info: Final model has been saved")
        self.train_end_time = time.time()
        elapsed_time = self.train_end_time - self.train_start_time
        print(f'Info: Training has been completed...')
        print(f'Info: Total Completion Time: {elapsed_time:.2f} seconds')

        return rewards_per_episode, epsilon_decay_per_episode, training_error, steps_per_episode, success_rate

    def test_agent(self, episodes):
        print(f'Info: Testing of the Agent has been started over the Maze Simulation...')
        print(f'Info: Source: {self.source} Destination: {self.destination}')

        # Test Code
        location = np.load('data/source_destination.npy')
        self.source, self.destination = location[0], location[1]
        rewards_per_episode = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        success_rate = np.zeros(episodes)
        for episode in range(episodes):
            state = self.reset()
            done, total_reward, step, success_status = False, 0, 0, 0
            while not done:
                self.update_display() if self.render else None
                action = self.dqn_model.act(state)
                next_state, reward, terminated, truncated, info = self.step(action)
                state = next_state
                step += 1
                done = terminated or truncated
                total_reward += reward
                if info['Success']:
                    success_status = 1
            print(f'Episode {episode + 1}/{episodes} - Steps: {step}, Reward: {total_reward:.2f}')
            rewards_per_episode[episode] = total_reward
            steps_per_episode[episode] = step
            success_rate[episode] = success_status
        print(f'Info: Testing has been completed...')

        return rewards_per_episode, steps_per_episode, success_rate

    def update_display(self):
        self.maze.screen.fill(self.maze.WHITE)
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                color = (0, 0, 0) if self.maze.maze[y, x] == 1 else (255, 255, 255)
                pygame.draw.rect(self.maze.screen, color,
                                 pygame.Rect(x * self.maze.cell_size, y * self.maze.cell_size,
                                             self.maze.cell_size, self.maze.cell_size))
        if self.path:
            for i in range(len(self.path) - 1):
                start_pos = (self.path[i][0] * self.maze.cell_size + self.maze.cell_size // 2,
                             self.path[i][1] * self.maze.cell_size + self.maze.cell_size // 2)
                end_pos = (self.path[i + 1][0] * self.maze.cell_size + self.maze.cell_size // 2,
                           self.path[i + 1][1] * self.maze.cell_size + self.maze.cell_size // 2)
                pygame.draw.line(self.maze.screen, (0, 0, 255), start_pos, end_pos, 3)

        pygame.draw.circle(self.maze.screen, (0, 255, 0),
                           (self.source[0] * self.maze.cell_size + self.maze.cell_size // 2,
                            self.source[1] * self.maze.cell_size + self.maze.cell_size // 2), 8)
        pygame.draw.circle(self.maze.screen, (255, 0, 255),
                           (self.destination[0] * self.maze.cell_size + self.maze.cell_size // 2,
                            self.destination[1] * self.maze.cell_size + self.maze.cell_size // 2), 8)
        pygame.draw.circle(self.maze.screen, (139, 69, 19),
                           (self.agent.position[0] * self.maze.cell_size + self.maze.cell_size // 2,
                            self.agent.position[1] * self.maze.cell_size + self.maze.cell_size // 2), 10)

        pygame.display.update()
        self.maze.clock.tick(self.fps)
