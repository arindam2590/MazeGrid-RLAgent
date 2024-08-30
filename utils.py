import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataVisualization:
    def __init__(self, params, episodes, rewards, steps, training_error=0.0, epsilon_decay=0.0):
        self.model = params['MODEL_NAME']
        self.fig_dir = params['FIG_DIR']
        self.n_episodes = episodes
        self.rewards = rewards
        self.steps = steps
        self.training_error = training_error
        self.epsilon_decay_per_episode = epsilon_decay

    def save_data(self, filename):
        filepath = self.fig_dir + '/' + filename
        df = pd.DataFrame({'Rewards': self.rewards,
                           'Steps': self.steps,
                           'Epsilon Decay': self.epsilon_decay_per_episode,
                           'Training Error': self.training_error})

        if not os.path.isfile(filepath):
            with pd.ExcelWriter(filepath, mode='w') as writer:
                df.to_excel(writer, sheet_name=self.model)

        else:
            with pd.ExcelWriter(filepath, mode='a') as writer:
                df.to_excel(writer, sheet_name=self.model)

    def plot_rewards(self, filename):
        """
        Plot the cumulative rewards over episodes.

        Parameters:
        - rewards: Array of rewards per episode.
        - filename: The name of the file to save the plot.
        - episodes: The number of episodes.
        """
        plot_filename = self.fig_dir + '/meshgrid_' + self.model + '_' + filename

        sum_rewards = np.zeros(self.n_episodes)

        for episode in range(self.n_episodes):
            sum_rewards[episode] = np.sum(self.rewards[0:(episode + 1)])

        # Plot and save the cumulative reward graph
        plt.plot(sum_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Cumulative Reward per Episodes')
        plt.savefig(plot_filename)
        plt.clf()

    # Function to plot episode length over episodes
    def plot_episode_length(self, filename):
        """
        Plot the length of episodes over time.

        Parameters:
        - steps: Array of steps taken per episode.
        - filename: The name of the file to save the plot.
        - episodes: The number of episodes.
        """
        plot_filename = self.fig_dir + '/meshgrid_' + self.model + '_' + filename
        plt.plot(range(self.n_episodes), self.steps)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Length per Episode')
        plt.savefig(plot_filename)
        plt.clf()

    # Function to plot epsilon decay over episodes
    def plot_epsilon_decay(self, filename):
        """
        Plot the decay of epsilon over episodes.

        Parameters:
        - epsilon_decay_per_episode: Array of epsilon values per episode.
        - filename: The name of the file to save the plot.
        - episodes: The number of episodes.
        """
        plot_filename = self.fig_dir + '/meshgrid_' + self.model + '_' + filename
        plt.plot(range(self.n_episodes), self.epsilon_decay_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.title('Epsilon Decay per Episode')
        plt.savefig(plot_filename)
        plt.clf()

    # Function to plot training error (temporal difference) over episodes
    def plot_training_error(self, filename):
        """
        Plot the temporal difference error over episodes.

        Parameters:
        - training_error: Array of temporal difference errors per episode.
        - filename: The name of the file to save the plot.
        - episodes: The number of episodes.
        """
        plot_filename = self.fig_dir + '/meshgrid_' + self.model + '_' + filename
        plt.plot(range(self.n_episodes), self.training_error)
        plt.xlabel('Episodes')
        plt.ylabel('Temporal Difference')
        plt.title('Temporal Difference per Episode')
        plt.savefig(plot_filename)
        plt.clf()
