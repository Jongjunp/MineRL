import minerl
import gym
import numpy as np
import math
import tensorflow.keras as keras
from collections import deque
import random

    #observation space: compassangle:array,
    # inventory:dict with dirt number,
    # pov:array(Box(width, height, nchannels))

    #action space: attack, back, forward, jump, left, right, place, sneak, sprint, camera[2]

NUM_ACTION_SPACE = 64

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

                 target_update_freq,

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
        self.target_update_freq = target_update_freq
        self.minibatch_size = minibatch_size
        self.minibatch_step_size = minibatch_step_size
        self.episode_num = episode_num

        self.main_model = self.intrinsic_create_model()
        self.target_model = self.intrinsic_create_model()
        self.target_model.set_weights(self.main_model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)

        self.target_update_counter = 0

        self.epsilon = epsilon_i

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
        model.add(keras.layers.Dense(96, kernel_initializer='normal', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(NUM_ACTION_SPACE, kernel_initializer='normal', activation='softmax'))

        model.compile(optimizer='rmsprop', loss=keras.losses.categorical_crossentropy())
        model.summary()

        return model

    #updating replay memory with infos:
    #[current state, current action, reward, next state, stop condition]
    def update_replay_memory(self, current_state, current_action, reward, next_state, done):
        self.replay_memory.append((current_state, current_action, reward, next_state, done))


    #training
    def train(self, done):

        #guarantee the minimum num of samples
        if len(self.replay_memory) < self.min_replay_size:
            return

        #decrease the value of epsilon when each episodes are ended
        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

        #mini batch sampling
        mini_batch = random.sample(self.replay_memory, self.minibatch_size)
        #mini batch construction
        current_states = np.stack([single_batch[0] for single_batch in mini_batch])
        current_actions = self.main_model.predict(current_states)
        next_states = np.stack([sample[3] for sample in mini_batch])
        next_actions = self.target_model.predict(next_states)

        for i, (current_state, current_action, reward, next_state, done) in enumerate(mini_batch):
            if done:
                next_action = reward
            else:
                next_action = reward + (self.discount_rate*np.max(next_actions[i]))
            current_actions[i, current_action] = next_action

        #fit model
        hist = self.main_model.fit(current_states, current_actions,
                                      batch_size=self.minibatch_size, verbose=0)
        loss = hist.history['loss'][0]
        return loss


    #determining action
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, NUM_ACTION_SPACE)
        else:
            predict = np.argmax(self.main_model.predict(state))
            return np.asscalar(predict)



    #increase target update counter & update target network if necessary
    def update_target_network(self):
        self.target_update_counter += 1
        if (self.target_update_counter >= self.target_update_freq):
            self.target_model.set_weights(self.main_model.get_weights())
            self.target_update_counter = 0







