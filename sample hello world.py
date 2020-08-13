import minerl
import gym
import numpy as np



if __name__ == '__main__':

    env = gym.make('MineRLNavigateDense-v0')


    obs  = env.reset()
    done = False
    net_reward = 0
    print(obs)

    #######################################
    while not done:
        action = env.action_space.noop()
        # action space = attack, back, forward, jump, left, right, place, sneak, sprint, camera

        action['camera'] = [0, 0.03 * obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print("Total reward: ", net_reward)
######################################
