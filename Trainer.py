import minerl
import gym
import numpy as np
import math
import tensorflow.keras as keras
from collections import deque
import random
import Agent

if __name__ == '__main__':

    env = gym.make("MineRLTreechopVectorObf-v0")  # A MineRLTreechopVectorObf-v0 env

    obs = env.reset()
    done = False

    while not done:
        obs, reward, done, act = env.step(env.action_space.noop())

    # Sample some data from the dataset!
    data = minerl.data.make("MineRLTreechopVectorObf-v0")

    # Iterate through a single epoch using sequences of at most 32 steps
    for obs, reward, done, act in data.seq_iter(num_epochs=1):
