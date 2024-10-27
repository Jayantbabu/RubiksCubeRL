import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import time as ti


print("TensorFlow version:", tf.__version__)

# List physical devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

# Confirm GPU usage
if len(physical_devices) > 0:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is not using the GPU.")

# Environment class for the Rubik's Cube
class RubiksCubeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize a solved cube state
        self.state = np.array([i // 9 for i in range(54)])
        return self.get_observation()

    def get_observation(self):
        # One-hot encode the cube state
        observation = np.zeros((54, 6))
        for i, color in enumerate(self.state):
            observation[i, int(color)] = 1
        return observation.flatten()

    def step(self, action):
        # Apply the action to the cube state
        self.state = self.rotate(self.state, action)
        reward = self.get_reward()
        done = reward == 1
        return self.get_observation(), reward, done

    def rotate(self, state, action):
        # Define rotation transformations for each action
        # Each action corresponds to a permutation of the cube's stickers
        # We will use predefined mappings for each move
        move_mappings = self.get_move_mappings()
        mapping = move_mappings[action]
        new_state = state[mapping]
        return new_state

    def get_move_mappings(self):
        # Define the mapping for each of the 12 moves
        move_mappings = {}

        # U (Up face clockwise rotation)
        move_mappings[0] = [
            6, 3, 0, 7, 4, 1, 8, 5, 2,  # Up face
            36, 37, 38, 12, 13, 14, 15, 16, 17,  # Left to Back
            18, 19, 20, 21, 22, 23, 9, 10, 11,  # Front to Left
            27, 28, 29, 30, 31, 32, 33, 34, 35,  # Right remains
            42, 43, 44, 39, 40, 41, 24, 25, 26,  # Back to Right
            45, 46, 47, 48, 49, 50, 51, 52, 53   # Down remains
        ]

        # U' (Up face counterclockwise rotation)
        move_mappings[1] = [
            2, 5, 8, 1, 4, 7, 0, 3, 6,  # Up face
            24, 25, 26, 12, 13, 14, 15, 16, 17,  # Left to Front
            18, 19, 20, 21, 22, 23, 42, 43, 44,  # Front to Right
            27, 28, 29, 30, 31, 32, 33, 34, 35,  # Right remains
            9, 10, 11, 39, 40, 41, 36, 37, 38,  # Back to Left
            45, 46, 47, 48, 49, 50, 51, 52, 53   # Down remains
        ]

        # Similarly define mappings for D, D', L, L', R, R', F, F', B, B'
        # D (Down face clockwise rotation)
        move_mappings[2] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8,  # Up remains
            9, 10, 11, 12, 13, 14, 51, 52, 53,  # Left to Front
            18, 19, 20, 21, 22, 23, 24, 25, 26,  # Front remains
            27, 28, 29, 30, 31, 32, 45, 46, 47,  # Right to Back
            33, 34, 35, 36, 37, 38, 39, 40, 41,  # Back remains
            48, 49, 50, 15, 16, 17, 42, 43, 44   # Down face rotates
        ]

        # D' (Down face counterclockwise rotation)
        move_mappings[3] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 48, 49, 50,
            18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 15, 16, 17,
            33, 34, 35, 36, 37, 38, 39, 40, 41,
            45, 46, 47, 51, 52, 53, 42, 43, 44
        ]

        # L (Left face clockwise rotation)
        move_mappings[4] = [
            18, 1, 2, 21, 4, 5, 24, 7, 8,  # Up to Front
            9, 10, 11, 12, 13, 14, 15, 16, 17,
            45, 19, 20, 48, 22, 23, 51, 25, 26,  # Down to Back
            0, 28, 29, 3, 31, 32, 6, 34, 35,  # Front to Down
            36, 37, 38, 39, 40, 41, 42, 43, 44,
            27, 46, 47, 30, 49, 50, 33, 52, 53   # Left face rotates
        ]

        # L' (Left face counterclockwise rotation)
        move_mappings[5] = [
            27, 1, 2, 30, 4, 5, 33, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17,
            0, 19, 20, 3, 22, 23, 6, 25, 26,
            45, 28, 29, 48, 31, 32, 51, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44,
            18, 46, 47, 21, 49, 50, 24, 52, 53
        ]

        # R (Right face clockwise rotation)
        move_mappings[6] = [
            0, 1, 36, 3, 4, 39, 6, 7, 42,
            9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 2, 21, 22, 5, 24, 25, 8,
            27, 28, 29, 30, 31, 32, 33, 34, 35,
            26, 37, 38, 23, 40, 41, 20, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53
        ]

        # R' (Right face counterclockwise rotation)
        move_mappings[7] = [
            0, 1, 20, 3, 4, 23, 6, 7, 26,
            9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 44, 21, 22, 41, 24, 25, 38,
            27, 28, 29, 30, 31, 32, 33, 34, 35,
            8, 37, 38, 5, 40, 41, 2, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53
        ]

        # F (Front face clockwise rotation)
        move_mappings[8] = list(range(54))  # Start with the identity mapping

        # Update the indices affected by the F move
        # Up face
        move_mappings[8][6], move_mappings[8][7], move_mappings[8][8] = 44, 41, 38
        # Left face
        move_mappings[8][24], move_mappings[8][25], move_mappings[8][26] = 6, 7, 8
        # Down face
        move_mappings[8][45], move_mappings[8][46], move_mappings[8][47] = 24, 25, 26
        # Right face
        move_mappings[8][38], move_mappings[8][41], move_mappings[8][44] = 47, 46, 45
        # Front face rotation (corners and edges)
        move_mappings[8][18], move_mappings[8][19], move_mappings[8][20], move_mappings[8][21], move_mappings[8][22], move_mappings[8][23], move_mappings[8][24], move_mappings[8][25], move_mappings[8][26] = \
            24, 21, 18, 25, 22, 19, 26, 23, 20

        # Corrected F' (Front face counterclockwise rotation)
        move_mappings[9] = list(range(54))  # Start with the identity mapping

        # Update the indices affected by the F' move
        # Up face
        move_mappings[9][6], move_mappings[9][7], move_mappings[9][8] = 24, 25, 26
        # Left face
        move_mappings[9][24], move_mappings[9][25], move_mappings[9][26] = 45, 46, 47
        # Down face
        move_mappings[9][45], move_mappings[9][46], move_mappings[9][47] = 44, 41, 38
        # Right face
        move_mappings[9][38], move_mappings[9][41], move_mappings[9][44] = 8, 7, 6
        # Front face rotation (corners and edges)
        move_mappings[9][18], move_mappings[9][19], move_mappings[9][20], move_mappings[9][21], move_mappings[9][22], move_mappings[9][23], move_mappings[9][24], move_mappings[9][25], move_mappings[9][26] = \
            20, 23, 26, 19, 22, 25, 18, 21, 24

        # B (Back face clockwise rotation)
        move_mappings[10] = [
            33, 34, 35, 3, 4, 5, 6, 7, 8,
            0, 1, 2, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 51, 52, 53,
            9, 10, 11, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50
        ]

        # B' (Back face counterclockwise rotation)
        move_mappings[11] = [
            9, 10, 11, 3, 4, 5, 6, 7, 8,
            42, 43, 44, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 0, 1, 2,
            33, 34, 35, 36, 37, 38, 39, 40, 41,
            45, 46, 47, 48, 49, 50, 51, 52, 53
        ]

        return move_mappings

    def get_reward(self):
        # Reward is +1 if the cube is solved
        correct_stickers = sum(self.state == (self.state // 9))
        total_stickers = 54
        reward = correct_stickers / total_stickers
        if self.is_solved():
            return 10
        else:
            return reward - 0.01  # Small penalty for each move

    def is_solved(self):
        # Check if all stickers on each face have the same color
        for i in range(0, 54, 9):
            if len(set(self.state[i:i+9])) != 1:
                return False
        return True

    def scramble(self, num_moves):
        # Scramble the cube with a number of random moves
        for _ in range(num_moves):
            action = random.randint(0, 11)
            self.state = self.rotate(self.state, action)
        return self.get_observation()

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        # Neural network to approximate Q-value function
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        # Experience replay to train the network
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                q_future = np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
                target = reward + self.gamma * q_future
            else:
                target_f = self.model.predict(state[np.newaxis, :], verbose=0)
                target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)
        if self.epsilon % 1000 == 0:
            self.update_target_network()

# Training loop
def train_agent(t):
    env = RubiksCubeEnv()
    state_size = 324  # 54 stickers * 6 colors
    action_size = 12  # 12 possible moves
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32
    max_moves = 40
    rewards = []

    for e in range(episodes):
        scramble_moves = random.randint(3, max_moves)
        state = env.scramble(scramble_moves)
        total_reward = 0
        for time in range(max_moves):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == max_moves - 1:
                print(f"Time: {ti.time() - t} :Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")
                rewards.append(total_reward)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    # Save the trained model
    agent.model.save('dqn_rubiks_cube.h5')

# Testing the trained model
def test_agent():
    env = RubiksCubeEnv()
    agent = DQNAgent(324, 12)
    agent.model = models.load_model('dqn_rubiks_cube.h5')
    scramble_moves = random.randint(3, 40)
    state = env.scramble(scramble_moves)
    print("Testing the agent...")
    for _ in range(40):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        if done:
            print("Cube solved!")
            break
    else:
        print("Failed to solve the cube.")

if __name__ == "__main__":
    t = ti.time()
    train_agent(t)
    test_agent()
