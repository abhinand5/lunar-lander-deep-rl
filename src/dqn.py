# Numerical Computing
import numpy as np 

# OpenAI Gym
import gym 

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Utilities
from collections import deque # For Replay Memory
from tqdm import tqdm
import os
import time 

class DeepQNetwork:
    def __init__(self, model_name, **kwargs):
        # Initialize class variables
        self.model_name = model_name
        self.input_dim = kwargs['input_dim']
        self.n_actions = kwargs['n_actions']
        self.layer1_units = kwargs['layer1_units']
        self.layer2_units = kwargs['layer2_units']
        self.lr = kwargs['lr']
    
    def create_model(self):
        # Define Model Architecture
        model = Sequential()
        model.add(Dense(self.layer1_units, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.layer2_units, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        # Compile Model
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        
        return model