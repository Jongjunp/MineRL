import minerl
import gym
import numpy as np
import math
import tensorflow.keras as keras
from collections import deque
import random
import DQNAgent

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





if __name__ == '__main__':

    env = gym.make("MineRLTreechopVectorObf-v0")  # A MineRLTreechopVectorObf-v0 env

    obs = env.reset()
    done = False

    agent = DQNAgent.Agent(REPLAY_MEMORY_SIZE, MIN_REPLAY_SIZE,
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

        if episode % PRINT_INTERVAL and episode!=0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f}"
                  .format(step, episode, np.mean(rewards), np.mean(losses)))
            rewards = []
            losses = []



