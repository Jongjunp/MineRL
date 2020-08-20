# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl
import numpy as np
import tensorflow.keras as keras
from collections import deque
import random
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
os.environ['MINERL_DATA_ROOT'] = 'C:\Users\J.J.Park\PycharmProjects\MineRL\data\MineRLObtainDiamondVectorObf-v0'
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

#hyperparameters setting
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_SIZE = 5000

RUN_EPISODE_NUM = 25000
MIN_EPISODE_NUM = 1000

EPSILON_I = 1.0
EPSILON_DECAY = 1/(RUN_EPISODE_NUM-MIN_EPISODE_NUM)
EPSILON_MIN = 0.1

LEARNING_RATE = 0.00025
DISCOUNT_RATE = 0.9

TARGET_UPDATE_FREQ = 10000

MINIBATCH_SIZE = 32

PRINT_INTERVAL = 100
#MINIBATCH_STEP_SIZE =

SAVE_FREQ = 1000
MAIN_MODEL_PATH = 'train/main_model'
TARGET_MODEL_PATH = 'train/target_model'

NUM_ACTION_SPACE = 64
LOW = -1.0499999523162842
HIGH = 1.0499999523162842

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

                 minibatch_size):
        self.replay_memory_size = replay_memory_size
        self.min_replay_size = min_replay_size
        self.epsilon_i = epsilon_i
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.target_update_freq = target_update_freq
        self.minibatch_size = minibatch_size

        self.main_model = self.intrinsic_create_model()
        self.target_model = self.intrinsic_create_model()
        self.target_model.set_weights(self.main_model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)

        self.target_update_counter = 0

        self.epsilon = epsilon_i

    #create CNN network for Q-value
    def intrinsic_create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv3D(8, 8, padding='same', input_shape=(64, 64, 64, 3), activation='relu'))
        model.add(keras.layers.MaxPool3D((2, 2, 2)))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Conv3D(16, 4, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool3D((2, 2, 2)))
        model.add(keras.layers.Conv3D(16, 2, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool3D((2, 2, 2)))
        model.add(keras.layers.Conv3D(16, 2, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool3D((2, 2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv3D(32, 2, padding='same', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(96, kernel_initializer='normal', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(NUM_ACTION_SPACE, kernel_initializer='normal', activation='softmax'))

        model.compile(optimizer='rmsprop', loss=keras.losses.CategoricalCrossentropy())
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
            rand_action = {
                'vector' : np.random.randint(low=LOW, high=HIGH, size=NUM_ACTION_SPACE)
            }
            return rand_action
        else:
            index = np.argmax(self.main_model.predict(state))
            trained_action = {
                'vector' : np.zeros((1,NUM_ACTION_SPACE))
            }
            trained_action[index] = HIGH
            return trained_action

    #increase target update counter & update target network if necessary
    def update_target_network(self):
        self.target_update_counter += 1
        if (self.target_update_counter >= self.target_update_freq):
            self.target_model.set_weights(self.main_model.get_weights())
            self.target_update_counter = 0

    def save(self,model_filepath,target_model_path):
        self.main_model.save(model_filepath)
        self.target_model.save(target_model_path)

    def load(self, model_filepath,target_model_path):
        self.main_model.save(model_filepath)
        self.target_model.save(target_model_path)

def main():
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    # Sample code for illustration, add your training code below
    env = gym.make(MINERL_GYM_ENV)
    #env.make_interactive(port=31415, realtime=True)

    #MY_CODE_BELOW_HERE
    obs = env.reset()
    done = False

    agent = Agent(REPLAY_MEMORY_SIZE, MIN_REPLAY_SIZE,
                           EPSILON_I, EPSILON_DECAY, EPSILON_MIN,
                           LEARNING_RATE, DISCOUNT_RATE,
                           TARGET_UPDATE_FREQ, MINIBATCH_SIZE)

    step = 0
    rewards = []
    losses = []

    #training for RUN_EPISODE_NUM
    for episode in range(RUN_EPISODE_NUM):

        state = obs
        episode_reward = 0
        done = False
        #one episode per a loop.
        while not done:
            step += 1

            #determine action and apply action in MineRl env
            action = agent.get_action(state)
            obs, rew, halt, _ = env.step(action)

            # information aquisition
            next_state = obs
            reward = rew
            episode_reward += reward
            done = halt

            #in training mode, store data in replay memory
            agent.update_replay_memory(state, action, reward, next_state, done)
            #state update(in this case 'pov')
            state = next_state

            if episode > MIN_EPISODE_NUM:
                loss = agent.train(done)
                losses.append(loss)

                #updating target network
                if step%TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()


        rewards.append(episode_reward)
        if step%SAVE_FREQ==0 and step!=0:
            agent.save(MAIN_MODEL_PATH, TARGET_MODEL_PATH)


#     actions = [env.action_space.sample() for _ in range(10)] # Just doing 10 samples in this example
#     xposes = []
#     for _ in range(1):
#         obs = env.reset()
#         done = False
#         netr = 0

#         # Limiting our code to 1024 steps in this example, you can do "while not done" to run till end
#         while not done:

            # To get better view in your training phase, it is suggested
            # to register progress continuously, example when 54% completed
            # aicrowd_helper.register_progress(0.54)

            # To fetch latest information from instance manager, you can run below when you want to know the state
            #>> parser.update_information()
            #>> print(parser.payload)
            # .payload: provide AIcrowd generated json
            # Example: {'state': 'RUNNING', 'score': {'score': 0.0, 'score_secondary': 0.0}, 'instances': {'1': {'totalNumberSteps': 2001, 'totalNumberEpisodes': 0, 'currentEnvironment': 'MineRLObtainDiamond-v0', 'state': 'IN_PROGRESS', 'episodes': [{'numTicks': 2001, 'environment': 'MineRLObtainDiamond-v0', 'rewards': 0.0, 'state': 'IN_PROGRESS'}], 'score': {'score': 0.0, 'score_secondary': 0.0}}}}
            # .current_state: provide indepth state information avaiable as dictionary (key: instance id)


    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    env.close()


if __name__ == "__main__":
    main()
