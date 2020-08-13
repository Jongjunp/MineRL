import minerl
import gym
import numpy as np
import math
import random
import tensorflow.keras as keras

    #observation space: compassangle:array,
    # inventory:dict with dirt number,
    # pov:array(Box(width, height, nchannels))

    #action space: attack, back, forward, jump, left, right, place, sneak, sprint, camera

class Agent:
    #set hyperparameters
    def __init__(self,
                 replay_memory_size,
                 min_replay_size,
                 epsilon_i,
                 epsilon_decay,
                 epsilon_min,
                 learning_rate,
                 node_num,
                 discount_rate,
                 minibatch_size,
                 minibatch_step_size,
                 episode_num):
        self.replay_memory_size = replay_memory_size
        self.min_replay_size = min_replay_size
        self.epsilon_i = epsilon_i
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.node_num = node_num
        self.discount_rate = discount_rate
        self.minibatch_size = minibatch_size
        self.minibatch_step_size = minibatch_step_size
        self.episode_num = episode_num

    #create CNN network for Q-value
    def create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense())

