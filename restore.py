import gym
import numpy as np
from gym import wrappers
from policyimport Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Scaler
import argparse
import signal
from MIP import MipEnv
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt

def main(num_episodes,kl_targ, hid1_mult, policy_logvar):
    env = MipEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_dim += 1

    #Load scaler and rewards
    f = open("models/scaler-"+str(num_episodes)+".pkl",'rb')
    f2 = open("models/rewards-"+str(num_episodes)+".pkl",'rb')
    unpickler = pickle.Unpickler(f)
    scaler = unpickler.load()
    rws = pickle.load(f2)

    #Plot average rewards
    episode_list = np.arange(0,num_episodes,100)
    plt.plot(episode_list,rws)
    plt.xlabel("Numero di episodi")
    plt.ylabel("Ricompensa media per time-step")
    plt.show()

    #Instantiate Policy object and restore weights
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
    if num_episodes>0:
        policy.restore(num_episodes)

    #Show animation
    while True:
        obs=env.reset()
        step = 0.0
        scale, offset = scaler.get()
        scale[-1] = 1.0  
        offset[-1] = 0.0  
        done = False
        while not done:
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  
            obs = (obs - offset) * scale  
            action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
            obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
            env.render1()
            env.render2()
            step += 1e-3 
    policy.close_sess()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Restore a pretrained model and statistics '
                                                  'and use it to display results '))

    parser.add_argument('-n', '--num_episodes', type = int, default=0)

    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)

    args = parser.parse_args()
    main(**vars(args))
