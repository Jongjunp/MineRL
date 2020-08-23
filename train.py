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
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
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
os.environ['MINERL_DATA_ROOT'] = 'data/'
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

EPSILON_I = 0
EPSILON_DECAY = 0.1/(RUN_EPISODE_NUM-MIN_EPISODE_NUM)
EPSILON_MIN = 0.1

LEARNING_RATE = 0.00025
DISCOUNT_RATE = 0.9

TARGET_UPDATE_FREQ = 10000

MINIBATCH_SIZE = 32

PRINT_FREQ = 10
#MINIBATCH_STEP_SIZE =

SAVE_FREQ = 1000
MAIN_MODEL_PATH = 'train/main_model'
TARGET_MODEL_PATH = 'train/target_model'

#constants for use
POV_INPUT_SHAPE = (64, 64, 3)
VECTOR_INPUT_SPACE = (64,)
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

    #create CNN network to get Q-value
    def intrinsic_create_model(self):

        pov_input = Input(shape=POV_INPUT_SHAPE, name='pov')
        conv2d_layer_1 = layers.Conv2D(8, 8, padding='same', input_shape=POV_INPUT_SHAPE, activation='relu')(pov_input)
        maxpool_layer_1 = layers.MaxPool2D((2, 2))(conv2d_layer_1)
        dropout_1 = layers.Dropout(0.1)(maxpool_layer_1)
        conv2d_layer_2 = layers.Conv2D(16, 4, padding='same', activation='relu')(dropout_1)
        maxpool_layer_2 = layers.MaxPool2D((2, 2))(conv2d_layer_2)
        dropout_2 = layers.Dropout(0.1)(maxpool_layer_2)
        conv2d_layer_3 = layers.Conv2D(16, 2, padding='same', activation='relu')(dropout_2)
        maxpool_layer_3 = layers.MaxPool2D((2, 2))(conv2d_layer_3)
        dropout_3 = layers.Dropout(0.2)(maxpool_layer_3)
        conv2d_layer_4 = layers.Conv2D(16, 2, padding='same', activation='relu')(dropout_3)
        maxpool_layer_4 = layers.MaxPool2D((2, 2))(conv2d_layer_4)
        dropout_4 = layers.Dropout(0.25)(maxpool_layer_4)
        conv2d_layer_5 = layers.Conv2D(32, 2, padding='same', activation='relu')(dropout_4)
        flatten = layers.Flatten()(conv2d_layer_5)
        pov_dense = layers.Dense(64, kernel_initializer='normal', activation='relu')(flatten)

        vector_input = Input(shape=VECTOR_INPUT_SPACE, name='vector')
        vector_dense_1 = layers.Dense(128, activation='relu')(vector_input)
        vector_dense_2 = layers.Dense(128, activation='relu')(vector_dense_1)

        concatenated = layers.concatenate([pov_dense, vector_dense_2])

        batch_normalized = layers.BatchNormalization()(concatenated)
        determined_action = layers.Dense(NUM_ACTION_SPACE, kernel_initializer='normal', activation='softmax')(batch_normalized)

        model = Model([pov_input, vector_input], determined_action)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
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
                'vector': np.random.uniform(low=LOW, high=HIGH, size=NUM_ACTION_SPACE)
            }
            return rand_action
        else:
            pov = np.reshape(np.array(state['pov']), (1, 64, 64, 3))
            vector = np.reshape(np.array(state['vector']), (1, 64))
            input_state = {'pov': pov, 'vector': vector}
            predicted_vector = self.main_model.predict(input_state)
            predicted_vector = np.ravel(predicted_vector).astype(np.float32)
            #temp_vector = predicted_vector-((np.max(predicted_vector)-np.min(predicted_vector))/2)
            #norm = (HIGH-LOW)/(np.max(predicted_vector)-np.min(predicted_vector))
            #result_vector = temp_vector * norm
            trained_action = {
                'vector': predicted_vector
            }
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
    data = minerl.data.make(environment=MINERL_GYM_ENV)

    # Sample code for illustration, add your training code below
    env = gym.make(MINERL_GYM_ENV)
    #env.make_interactive(port=31415, realtime=True)

    #MY_CODE_BELOW_HERE
    current_state = env.reset()
    done = False

    agent = Agent(REPLAY_MEMORY_SIZE, MIN_REPLAY_SIZE,
                           EPSILON_I, EPSILON_DECAY, EPSILON_MIN,
                           LEARNING_RATE, DISCOUNT_RATE,
                           TARGET_UPDATE_FREQ, MINIBATCH_SIZE)

    step = 0
    episode_reward = 0
    rewards = []
    losses = []

    aicrowd_helper.training_start()

    #one episode per a loop.
    while not done:

        step += 1
        episode_reward = 0

        #determine action and apply action in MineRl env
        action = agent.get_action(current_state)
        next_state, reward, done, _ = env.step(action)

        # information aquisition
        episode_reward += reward

        #in training mode, store data in replay memory
        agent.update_replay_memory(current_state, action, reward, next_state, done)
        #state update(in this case 'pov')
        current_state = next_state

        loss = agent.train(done)
        losses.append(loss)

        #updating target network
        if step%TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

    rewards.append(episode_reward)


    agent.save(MAIN_MODEL_PATH, TARGET_MODEL_PATH)


    print("step: {} / reward: {:.2f} / loss: {:.4f}"
        .format(step, np.mean(rewards), np.mean(losses)))


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
