import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        minibatch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        minibatch = []
        for idx in minibatch_indices:
            state, action, reward, next_state, done = self.buffer[idx]
            minibatch.append((state, action, reward, next_state, done))
        return minibatch


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience, error):
        self.buffer.append(experience)
        priority = (abs(error) + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = np.array(self.priorities) ** (1 - beta)
        probs = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-Sampling Weights
        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority


class DQNModel:
    def __init__(self, state_size, action_size, maze, is_training, model_path, params):
        self.state_size = state_size
        self.action_size = action_size
        self.maze = maze
        self.buffer_size = params['BUFFER_SIZE']
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.gamma = params['GAMMA']  # discount rate
        self.alpha = params['ALPHA']  # learning rate
        self.epsilon = params['EPSILON']  # exploration rate
        self.epsilon_min = params['EPSILON_MIN']
        self.epsilon_decay = params['EPSILON_DECAY']
        self.update_rate = params['UPDATE_RATE']
        self.main_network = self._build_network() if is_training else tf.keras.models.load_model(
                                    model_path + '/'+params['MODEL_NAME']+'_model_final.keras')
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def _build_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.state_size*2, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(self.state_size, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        Q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(Q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self, batch_size):
        sum = 0
        minibatch = self.replay_buffer.sample(batch_size)
        for sample in minibatch:
            state, action, reward, next_state, done = sample
            if not done:
                target_Q = reward + self.gamma * np.amax(self.target_network.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0])
            else:
                target_Q = reward
            Q_values = self.main_network.predict(np.reshape(state, [1, self.state_size]), verbose=0)
            sum += np.square(target_Q - np.amax(Q_values))
            Q_values[0][action] = target_Q
            self.main_network.fit(np.reshape(state, [1, self.state_size]), Q_values, epochs=1, verbose=0)

        loss = sum/batch_size

        return loss


class DoubleDQNModel:
    def __init__(self, state_size, action_size, maze, is_training, model_path, params):
        self.state_size = state_size
        self.action_size = action_size
        self.maze = maze
        self.buffer_size = params['BUFFER_SIZE']
        self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size)
        self.gamma = params['GAMMA']  # discount rate
        self.alpha = params['ALPHA']  # learning rate
        self.epsilon = params['EPSILON']  # exploration rate
        self.epsilon_min = params['EPSILON_MIN']
        self.epsilon_decay = params['EPSILON_DECAY']
        self.update_rate = params['UPDATE_RATE']
        self.main_network = self._build_network() if is_training else tf.keras.models.load_model(
                                                                                    model_path + '/model_final.keras')
        self.target_network = self._build_network() if is_training else None
        self.target_network.set_weights(self.main_network.get_weights())

    def _build_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.state_size*2, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(self.state_size, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        Q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(Q_values[0])

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        Q_values = self.main_network.predict(state, verbose=0)
        target_Q = reward
        if not done:
            next_action = np.argmax(self.main_network.predict(next_state, verbose=0)[0])
            target_Q += self.gamma * self.target_network.predict(next_state, verbose=0)[0][next_action]
        error = target_Q - Q_values[0][action]
        self.replay_buffer.add((state, action, reward, next_state, done), error)

    def train(self, batch_size):
        if len(self.replay_buffer.buffer) < batch_size:
            return 0

        minibatch, indices, weights = self.replay_buffer.sample(batch_size)
        errors = np.zeros(batch_size)
        sum_loss = 0

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])

            target_Q = reward
            if not done:
                next_action = np.argmax(self.main_network.predict(next_state, verbose=0)[0])
                target_Q += self.gamma * self.target_network.predict(next_state, verbose=0)[0][next_action]

            Q_values = self.main_network.predict(state, verbose=0)
            errors[i] = target_Q - Q_values[0][action]
            Q_values[0][action] = target_Q

            self.main_network.fit(state, Q_values, sample_weight=np.array([weights[i]]), epochs=1, verbose=0)
            sum_loss += np.square(errors[i])


class DuelingDQNModel:
    def __init__(self, state_size, action_size, maze, is_training, model_path, params):
        self.state_size = state_size
        self.action_size = action_size
        self.maze = maze
        self.buffer_size = params['BUFFER_SIZE']
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.gamma = params['GAMMA']  # discount rate
        self.alpha = params['ALPHA']  # learning rate
        self.epsilon = params['EPSILON']  # exploration rate
        self.epsilon_min = params['EPSILON_MIN']
        self.epsilon_decay = params['EPSILON_DECAY']
        self.update_rate = params['UPDATE_RATE']
        self.main_network = self._build_network() if is_training else tf.keras.models.load_model(
            model_path + '/model_final.keras')
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def _build_network(self):
        inputs = tf.keras.Input(shape=(self.state_size,))
        shared = layers.Dense(self.state_size * 2, activation='relu')(inputs)
        shared = layers.Dense(self.state_size, activation='relu')(shared)

        # Value stream
        value = layers.Dense(1, activation='linear')(shared)

        # Advantage stream
        advantage = layers.Dense(self.action_size, activation='linear')(shared)

        # Combine value and advantage to get Q-values
        advantage_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        Q_values = layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])

        model = tf.keras.Model(inputs=inputs, outputs=Q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')

        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        Q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(Q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self, batch_size):
        sum = 0
        minibatch = self.replay_buffer.sample(batch_size)
        for sample in minibatch:
            state, action, reward, next_state, done = sample
            if not done:
                target_Q = reward + self.gamma * np.amax(self.target_network.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0])
            else:
                target_Q = reward
            Q_values = self.main_network.predict(np.reshape(state, [1, self.state_size]), verbose=0)
            sum += np.square(target_Q - np.amax(Q_values))
            Q_values[0][action] = target_Q
            self.main_network.fit(np.reshape(state, [1, self.state_size]), Q_values, epochs=1, verbose=0)

        loss = sum/batch_size

        return loss


class DRQNModel:
    def __init__(self, state_size, action_size, maze, is_training, model_path, params):
        self.state_size = state_size
        self.action_size = action_size
        self.maze = maze
        self.buffer_size = params['BUFFER_SIZE']
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.gamma = params['GAMMA']  # discount rate
        self.alpha = params['ALPHA']  # learning rate
        self.epsilon = params['EPSILON']  # exploration rate
        self.epsilon_min = params['EPSILON_MIN']
        self.epsilon_decay = params['EPSILON_DECAY']
        self.update_rate = params['UPDATE_RATE']
        self.time_steps = params['TIME_STEPS']  # Number of time steps for LSTM

        self.main_network = self._build_network() if is_training else tf.keras.models.load_model(
            model_path + '/model_final.keras')
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def _build_network(self):
        model = tf.keras.Sequential()
        model.add(layers.LSTM(64, input_shape=(self.time_steps, self.state_size), return_sequences=True))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        return model

    def act(self, state_sequence):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        # Ensure state_sequence is reshaped correctly
        if len(state_sequence) < self.time_steps:
            # If state_sequence is shorter than time_steps, pad it with zeros
            state_sequence = np.pad(state_sequence, ((self.time_steps - len(state_sequence), 0), (0, 0)), 'constant')
        elif len(state_sequence) > self.time_steps:
            # If state_sequence is longer than time_steps, truncate it
            state_sequence = state_sequence[-self.time_steps:]

        state_sequence = np.reshape(state_sequence, (1, self.time_steps, self.state_size))
        Q_values = self.main_network.predict(state_sequence, verbose=0)
        return np.argmax(Q_values[0])

    def remember(self, state_sequence, action, reward, next_state_sequence, done):
        self.replay_buffer.add((state_sequence, action, reward, next_state_sequence, done))

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self, batch_size):
        sum_loss = 0
        minibatch = self.replay_buffer.sample(batch_size)
        for state_sequence, action, reward, next_state_sequence, done in minibatch:
            # Ensure sequences are of consistent length
            if len(state_sequence) < self.time_steps:
                state_sequence = np.pad(state_sequence, ((self.time_steps - len(state_sequence), 0), (0, 0)), 'constant')
            elif len(state_sequence) > self.time_steps:
                state_sequence = state_sequence[-self.time_steps:]

            if len(next_state_sequence) < self.time_steps:
                next_state_sequence = np.pad(next_state_sequence, ((self.time_steps - len(next_state_sequence), 0), (0, 0)), 'constant')
            elif len(next_state_sequence) > self.time_steps:
                next_state_sequence = next_state_sequence[-self.time_steps:]

            state_sequence = np.reshape(state_sequence, [1, self.time_steps, self.state_size])
            next_state_sequence = np.reshape(next_state_sequence, [1, self.time_steps, self.state_size])

            if not done:
                target_Q = reward + self.gamma * np.amax(
                    self.target_network.predict(next_state_sequence, verbose=0)[0])
            else:
                target_Q = reward

            Q_values = self.main_network.predict(state_sequence, verbose=0)
            sum_loss += np.square(target_Q - np.amax(Q_values))
            Q_values[0][action] = target_Q
            self.main_network.fit(state_sequence, Q_values, epochs=1, verbose=0)

        loss = sum_loss / batch_size
        return loss

