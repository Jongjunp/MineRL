import minerl
import gym
import numpy as np
import math
import random
import tensorflow.keras as keras

    #observation space: compassangle:array,
    # inventory:dict with dirt number,
    # pov:array(Box(width, height, nchannels))

    #action space: attack, back, forward, jump, left, right, place, sneak, sprint, camera[2]

NUM_ACTION_SPACE = 10

class Agent:
    #set hyperparameters
    def __init__(self,
                 replay_memory_size,
                 min_replay_size,
                 epsilon_i,
                 epsilon_decay,
                 epsilon_min,
                 learning_rate,
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
        self.discount_rate = discount_rate
        self.minibatch_size = minibatch_size
        self.minibatch_step_size = minibatch_step_size
        self.episode_num = episode_num

        self.main_model = self.intrinsic_create_model()
        self.target_model = self.intrinsic_create_model()
        self.target_model.set_weights(self.main_model.get_weights())

    #create CNN network for Q-value
    def intrinsic_create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(8, (8, 8), padding='same', input_shape=(64, 64, 3), activation='relu'))
        model.add(keras.layers.MaxPool2D(2,2))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Conv2D(16, (4, 4), padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(2, 2))
        model.add(keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(2, 2))
        model.add(keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(2, 2))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(24, kernel_initializer='normal', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(NUM_ACTION_SPACE, kernel_initializer='normal', activation='softmax'))

        model.compile(optimizer='rmsprop', loss=keras.losses.categorical_crossentropy())

        return model

    #

