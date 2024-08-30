import json
import numpy as np
from maze_simulation import Simulation
from utils import DataVisualization


def main():
    with open('config.json', 'r') as file:
        params = json.load(file)

    train_mode = True
    is_trained = False
    is_test_completed = False
    render = False
    train_episodes = 10000
    test_episodes = 50

    sim = Simulation(params, train_mode, render)
    sim.game_initialize(params)
    while sim.running:
        sim.event_on_game_window() if render else None
        if train_mode and not is_trained:
            print(f'*' * 25 + ' Training Phase ' + '*' * 25)

            plot_filename = 'training_reward.png'
            plot_episode_len_filename = 'training_episode_len.png'
            decay_plot_filename = 'epsilon_decay.png'
            error_plot_filename = 'training_error.png'

            rewards, epsilon_decay, training_error, steps, success_rate = sim.train_agent(train_episodes)

            train_data_visual = DataVisualization(params, train_episodes, rewards, steps, training_error, epsilon_decay)

            excel_filename = 'Training_data.xlsx'
            train_data_visual.save_data(excel_filename)
            train_data_visual.plot_rewards(plot_filename)
            train_data_visual.plot_episode_length(plot_episode_len_filename)
            train_data_visual.plot_epsilon_decay(decay_plot_filename)
            train_data_visual.plot_training_error(error_plot_filename)

            is_trained = True

        if (is_trained or not train_mode) and not is_test_completed:
            print(f'*' * 25 + ' Testing Phase ' + '*' * 25)
            plot_filename = 'testing_reward.png'
            plot_episode_len_filename = 'testing_episode_len.png'

            rewards, steps, success_rate = sim.test_agent(test_episodes)
            test_data_visual = DataVisualization(params, test_episodes, rewards, steps)

            excel_filename = 'Testing_data.xlsx'
            test_data_visual.save_data(excel_filename)
            test_data_visual.plot_rewards(plot_filename)
            test_data_visual.plot_episode_length(plot_episode_len_filename)
            success = 100 * np.mean(success_rate)
            print('Test Success Rate:', success)

            is_test_completed = True
            break

    sim.close_window()


if __name__ == '__main__':
    main()
