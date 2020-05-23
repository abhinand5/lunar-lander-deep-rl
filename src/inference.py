# Numerical Computing
import numpy as np 

# OpenAI Gym
import gym 

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Custom imports
from custom_tensorboards import ModifiedTensorBoard
from dqn import DeepQNetwork

# Utilities
from collections import deque # For Replay Memory
from tqdm import tqdm
import os
import time 
import imageio 
import random

# TF Specific 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print(f"GPU in use: {tf.config.list_physical_devices('GPU')}")


# SECTION 1: 
# GLOBAL VARIABLES AND KEY INITIALIZATIONS
# =========================================================================================================

# Important
MODEL_NAME = 'DQN-Dense'
DISCOUNT = 0.99 # Our discount factor
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -100  # For model save
MEMORY_FRACTION = 0.20
LEARNING_RATE = 0.01

# Environment settings
env = gym.make('LunarLander-v2')
EPISODES = 5

# Model params
INPUT_DIM = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
LAYER1_UNITS = 512
LAYER2_UNITS = 256
LOAD_MODEL = "../models/DenseNet___304.03max__203.19avg__-75.29min__1589481774.model"

# Exploration settings
EPSILON_DECAY = 0.995 # Decay rate
MIN_EPSILON = 0.001 
epsilon = MIN_EPSILON

#  Stats settings
SHOW_PREVIEW = True
SHOW_EVERY = 1

# For stats
ep_rewards = [-100]

# For more atleast some degree of reproducibility
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# TO save renders
images = []
rdir = '../results'

# Create models folder
if not os.path.isdir('../models'):
    os.makedirs('../models')

# Create models folder
if not os.path.isdir('../results'):
    os.makedirs('../results')

# END SECTION 1

# ==========================================================================================================

# SECTION 2:
# IMPLEMENTATION OF THE RL AGENT
class DQNAgent:
    def __init__(self):
        self.dqn = DeepQNetwork(model_name=MODEL_NAME, 
                                input_dim=INPUT_DIM, 
                                n_actions=N_ACTIONS,
                                layer1_units=LAYER1_UNITS,
                                layer2_units=LAYER2_UNITS,
                                lr=LEARNING_RATE)
        # Our Main Model: POLICY NETWORK
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            self.model = load_model(LOAD_MODEL)
            print(f"Loaded Model: {LOAD_MODEL}")
        else:
            self.model = self.dqn.create_model()

        # TARGET NETWORK
        self.target_model = self.dqn.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # target update counter
        self.target_update_counter = 0

    
    def update_replay_memory(self, transition):
        ''' To update the replay with the steps' experience '''
        self.replay_memory.append(transition)


    def get_q_values(self, state):
        ''' Get the Q values (learned or thus far) '''
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

 
    def train(self, terminal_state, step):
        ''' This is where we actually train the Agent '''

        # Start training only if certain number of samples is already saved in REPLAY MEMORY
        # Else it keeps making steps which are added to the REPLAY MEM
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query Target NN model for Q values
        # Computing the max term in Bellman Equation
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), 
                       np.array(y), 
                       batch_size=MINIBATCH_SIZE, 
                       verbose=0, 
                       shuffle=False)

        # Update target network counter every episode
        if terminal_state: self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

# END : IMPLEMENTED THE AGENT

# ========================================================================================================

# USING THE AGENT

agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from learned Qs
            action = np.argmax(agent.get_q_values(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        # track the cumulative sum of episode rewards
        episode_reward += reward

        if SHOW_PREVIEW and not episode % SHOW_EVERY:
            img = env.render(mode='rgb_array')
            images.append(img)

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        # Transition to next state
        current_state = new_state
        step += 1

    # Last but not the least!
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

env.close()
imageio.mimsave(f'{rdir}/DQN-lander-{int(time.time())}.gif', [np.array(img) for i, img in enumerate(images)], fps=30)