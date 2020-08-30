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
import tqdm

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Input

from collections import deque
import random
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

#To solve the GPU memory problem
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechopVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
os.environ['MINERL_DATA_ROOT'] = 'data/MineRLObtainDiamondVectorObf-v0'
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/MineRLObtainDiamondVectorObf-v0')

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
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_SIZE = 500

RUN_EPISODE_NUM = 20000
MIN_EPISODE_NUM = 1000

EPSILON_I = 1
EPSILON_DECAY = 0.1/(RUN_EPISODE_NUM-MIN_EPISODE_NUM)
EPSILON_MIN = 0.1

ACTOR_LEARNING_RATE = 0.00075
CRITIC_LEARNING_RATE = 0.0015
DISCOUNT_RATE = 0.99
UPDATE_RATE = 0.001

ACTOR_OPTIMIZER = optimizers.RMSprop(ACTOR_LEARNING_RATE)
CRITIC_OPTIMIZER = optimizers.RMSprop(CRITIC_LEARNING_RATE)

MINIBATCH_SIZE = 32

PRINT_FREQ = 10
SAVE_FREQ = 1000
ACTOR_MODEL_PATH = 'train/actor_model'
TARGET_ACTOR_MODEL_PATH = 'train/target_actor_model'
CRITIC_MODEL_PATH = 'train/critic_model'
TARGET_CRITIC_MODEL_PATH = 'train/target_critic_model'

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

                 discount_rate,

                 minibatch_size):
        self.replay_memory_size = replay_memory_size
        self.min_replay_size = min_replay_size
        self.epsilon_i = epsilon_i
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_rate = discount_rate
        self.minibatch_size = minibatch_size

        self.actor = self._create_actor()
        self.target_actor = self._create_actor()

        self.critic = self._create_critic()
        self.target_critic = self._create_critic()

        #duplicating the weights of actor and critic into target models
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)

        self.target_update_counter = 0

        self.epsilon = epsilon_i

    # Actor class -> through the state, make the model for action vector
    def _create_actor(self):
        pov_input = Input(shape=POV_INPUT_SHAPE, name='pov')
        #normalize the input as having values between -1~1
        norm_pov_input = (pov_input-(255.0/2))-(255.0/2)

        conv2d_layer_1 = layers.Conv2D(8, 8, padding='same', input_shape=POV_INPUT_SHAPE, activation='relu')(norm_pov_input)
        maxpool_layer_1 = layers.MaxPool2D((2, 2))(conv2d_layer_1)
        dropout_1 = layers.Dropout(0.1)(maxpool_layer_1)
        conv2d_layer_2 = layers.Conv2D(16, 4, padding='same', activation='relu')(dropout_1)
        maxpool_layer_2 = layers.MaxPool2D((2, 2))(conv2d_layer_2)
        dropout_2 = layers.Dropout(0.1)(maxpool_layer_2)
        conv2d_layer_3 = layers.Conv2D(16, 2, padding='same', activation='relu')(dropout_2)
        maxpool_layer_3 = layers.MaxPool2D((2, 2))(conv2d_layer_3)
        dropout_3 = layers.Dropout(0.2)(maxpool_layer_3)
        #conv2d_layer_4 = layers.Conv2D(16, 2, padding='same', activation='relu')(dropout_3)
        #maxpool_layer_4 = layers.MaxPool2D((2, 2))(conv2d_layer_4)
        #dropout_4 = layers.Dropout(0.25)(maxpool_layer_4)
        conv2d_layer_5 = layers.Conv2D(32, 2, padding='same', activation='relu')(dropout_3)
        flatten = layers.Flatten()(conv2d_layer_5)
        pov_dense = layers.Dense(64, kernel_initializer='normal', activation='relu')(flatten)

        vector_input = Input(shape=VECTOR_INPUT_SPACE, name='vector')
        vector_dense_1 = layers.Dense(128, activation='relu')(vector_input)
        vector_dense_2 = layers.Dense(128, activation='relu')(vector_dense_1)

        concatenated = layers.concatenate([pov_dense, vector_dense_2])

        batch_normalized = layers.BatchNormalization()(concatenated)
        determined_action = layers.Dense(NUM_ACTION_SPACE, kernel_initializer='normal', activation='tanh')(batch_normalized)

        determined_action = determined_action * HIGH
        self.model = Model([pov_input, vector_input], determined_action)
        self.model.summary()

        return self.model

    # Critic class -> based on state and action, get Q-value
    def _create_critic(self):
        pov_input = Input(shape=POV_INPUT_SHAPE, name='pov')
        #normalize the input as having values between -1~1
        norm_pov_input = (pov_input-(255.0/2))-(255.0/2)

        conv2d_layer_1 = layers.Conv2D(8, 8, padding='same', input_shape=POV_INPUT_SHAPE, activation='relu')(norm_pov_input)
        maxpool_layer_1 = layers.MaxPool2D((2, 2))(conv2d_layer_1)
        dropout_1 = layers.Dropout(0.1)(maxpool_layer_1)
        conv2d_layer_2 = layers.Conv2D(16, 4, padding='same', activation='relu')(dropout_1)
        maxpool_layer_2 = layers.MaxPool2D((2, 2))(conv2d_layer_2)
        dropout_2 = layers.Dropout(0.1)(maxpool_layer_2)
        conv2d_layer_3 = layers.Conv2D(16, 2, padding='same', activation='relu')(dropout_2)
        maxpool_layer_3 = layers.MaxPool2D((2, 2))(conv2d_layer_3)
        dropout_3 = layers.Dropout(0.2)(maxpool_layer_3)
        #conv2d_layer_4 = layers.Conv2D(16, 2, padding='same', activation='relu')(dropout_3)
        #maxpool_layer_4 = layers.MaxPool2D((2, 2))(conv2d_layer_4)
        #dropout_4 = layers.Dropout(0.25)(maxpool_layer_4)
        conv2d_layer_5 = layers.Conv2D(32, 2, padding='same', activation='relu')(dropout_3)
        flatten = layers.Flatten()(conv2d_layer_5)
        pov_dense = layers.Dense(64, kernel_initializer='normal', activation='relu')(flatten)

        vector_input = Input(shape=VECTOR_INPUT_SPACE, name='vector')
        vector_dense_1 = layers.Dense(128, activation='relu')(vector_input)
        vector_dense_2 = layers.Dense(128, activation='relu')(vector_dense_1)

        concatenated = layers.concatenate([pov_dense, vector_dense_2])

        batch_normalized = layers.BatchNormalization()(concatenated)
        determined_action = layers.Dense(NUM_ACTION_SPACE, kernel_initializer='normal', activation='relu')(batch_normalized)

        action_input = Input(shape=VECTOR_INPUT_SPACE, name='action')
        action_dense_1 = layers.Dense(128, activation='relu')(action_input)

        concatenated_2 = layers.concatenate([determined_action, action_dense_1])

        action_dense_3 = layers.Dense(128, activation='relu')(concatenated_2)
        normalized_1 = layers.BatchNormalization()(action_dense_3)
        action_dense_4 = layers.Dense(128, activation='relu')(normalized_1)
        normalized_2 = layers.BatchNormalization()(action_dense_4)
        predicted_q = layers.Dense(1, activation=None)(normalized_2)

        self.model = Model([pov_input, vector_input, action_input], predicted_q)
        self.model.summary()

        return self.model


    #updating replay memory with infos:
    #[current state, current action, reward, next state, stop condition]
    def update_replay_memory(self, current_state, current_action, reward, next_state, done):
        self.replay_memory.append((current_state, current_action, reward, next_state, done))


    #training (adapting policy gradient: Actor-critic)
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

        #mini batch construction & mini batch learning
        #current_input preprocessing from current_state
        current_povs = np.reshape(np.stack([single_batch[0]['pov'] for single_batch in mini_batch])
                          ,(MINIBATCH_SIZE, 64, 64, 3))
        current_vectors = np.reshape(np.stack([single_batch[0]['vector'] for single_batch in mini_batch])
                          ,(MINIBATCH_SIZE, 64))
        current_states = {'pov': current_povs, 'vector': current_vectors}

        #next_input preprocessing from next_state
        next_povs = np.reshape(np.stack([single_batch[3]['pov'] for single_batch in mini_batch])
                          ,(MINIBATCH_SIZE, 64, 64, 3))
        next_vectors = np.reshape(np.stack([single_batch[3]['vector'] for single_batch in mini_batch])
                          ,(MINIBATCH_SIZE, 64))
        next_states = {'pov': next_povs, 'vector': next_vectors}

        #current_actions
        current_actions = np.reshape(np.stack([single_batch[1]['vector'] for single_batch in mini_batch])
                             ,(MINIBATCH_SIZE, 64))

        #rewards
        rewards = np.reshape(np.stack([single_batch[2] for single_batch in mini_batch])
                             ,(MINIBATCH_SIZE))

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_critic_input = {'pov': next_povs, 'vector': next_vectors, 'action': target_actions}
            critic_input = {'pov': current_povs, 'vector': current_vectors, 'action': current_actions}
            y = rewards + (self.discount_rate * self.target_critic(target_critic_input))
            critic_value_for_critic = self.critic(critic_input)
            critic_loss = tf.math.reduce_mean(tf.math.square(y-critic_value_for_critic))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        CRITIC_OPTIMIZER.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(current_states)
            critic_input_for_actor = {'pov': current_povs, 'vector': current_vectors, 'action': actions}
            critic_value_for_actor = self.critic(critic_input_for_actor)
            actor_loss = -tf.math.reduce_mean(critic_value_for_actor)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        ACTOR_OPTIMIZER.apply_gradients(zip(actor_grad, self.actor.trainable_variables))


    #determining action -> it is ddpg but use epsilon-greedy first
    def get_action(self, state, sample_actions):
        if self.epsilon > np.random.rand():
            sample_action = random.choice(sample_actions)
            action = {
                'vector': sample_action
            }
            return action
        else:
            pov = np.reshape(np.array(state['pov']), (1, 64, 64, 3))
            vector = np.reshape(np.array(state['vector']), (1, 64))
            input_state = {'pov': pov, 'vector': vector}
            predicted_vector = self.actor(input_state)
            predicted_vector = np.ravel(predicted_vector).astype(np.float32)
            predicted_vector = np.clip(predicted_vector, LOW, HIGH)
            trained_action = {
                'vector': predicted_vector
            }
            return trained_action

    #soft updating target network
    def update_target_network(self, update_rate):

        new_actor_weights = []
        new_critic_weights = []

        target_actor_variables = self.target_actor.get_weights()
        for i, var in enumerate(self.actor.get_weights()):
            new_actor_weights.append((var * update_rate) +
                                     (target_actor_variables[i]*(1-update_rate)))

        self.target_actor.set_weights(new_actor_weights)

        target_critic_variables = self.target_critic.get_weights()
        for i, var in enumerate(self.critic.get_weights()):
            new_critic_weights.append((var * update_rate) +
                                      (target_critic_variables[i]*(1-update_rate)))

        self.target_critic.set_weights(new_critic_weights)


    def save(self,actor_path,target_actor_path,critic_path,target_critic_path):
        self.actor.save(actor_path)
        self.target_actor.save(target_actor_path)
        self.critic.save(critic_path)
        self.target_critic.save(target_critic_path)

    def load(self,actor_path,target_actor_path,critic_path,target_critic_path):
        self.actor = models.load_model(actor_path)
        self.target_actor = models.load_model(target_actor_path)
        self.critic = models.load_model(critic_path)
        self.target_critic = models.load_model(target_critic_path)

def main():
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    data = minerl.data.make(environment=MINERL_GYM_ENV)

    # Sample code for illustration, add your training code below
    env = gym.make(MINERL_GYM_ENV)
    env.make_interactive(port=6666, realtime=True)

    #MY_CODE_BELOW_HERE
    done = False

    agent = Agent(REPLAY_MEMORY_SIZE, MIN_REPLAY_SIZE,
                  EPSILON_I, EPSILON_DECAY, EPSILON_MIN,
                  DISCOUNT_RATE,
                  MINIBATCH_SIZE)

    step = 0
    rewards = []

    # renewing act_vectors to maintain a variety of action
    act_vectors = []
    for _, act, _, _, _ in tqdm.tqdm(data.batch_iter(32, 16, 1)):
        act_vectors.append(act['vector'])
        if len(act_vectors) > 1000:
            break
    random.shuffle(act_vectors)
    acts = np.concatenate(act_vectors).reshape(-1, 64)

    aicrowd_helper.training_start()

    for episode in range(RUN_EPISODE_NUM):

        current_state = env.reset()
        done = False
        episode_reward = 0

        #one episode per a loop.
        while not done:

            step += 1
            #determine action and apply action in MineRl env
            action = agent.get_action(current_state, acts)
            next_state, reward, done, _ = env.step(action)

            # information aquisition
            episode_reward += reward

            #in training mode, store data in replay memory
            agent.update_replay_memory(current_state, action, reward, next_state, done)
            #training model
            agent.train(done)
            #soft updating target network
            agent.update_target_network(UPDATE_RATE)

            #state update
            current_state = next_state

        rewards.append(episode_reward)

        # renewing act_vectors to maintain a variety of action
        act_vectors.clear()
        for _, act, _, _, _ in tqdm.tqdm(data.batch_iter(32, 16, 1)):
            act_vectors.append(act['vector'])
            if len(act_vectors) > 1000:
                break
        random.shuffle(act_vectors)
        acts = np.concatenate(act_vectors).reshape(-1, 64)

        if episode%SAVE_FREQ==0 and episode!=0:
            agent.save(ACTOR_MODEL_PATH, TARGET_ACTOR_MODEL_PATH,
                       CRITIC_MODEL_PATH, TARGET_CRITIC_MODEL_PATH)
            print("Save model {}".format(episode))

        if episode%PRINT_FREQ==0 and episode!=0:
            print("step: {} / reward: {:.2f}"
                  .format(step, np.mean(rewards)))


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
