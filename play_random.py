"""A simple script for debugging the Super Mario Bros. Lua code."""
from time import sleep
from tqdm import tqdm
import gym
import nesgym_super_mario_bros


try:
    env = gym.make('nesgym/SuperMarioBros-v0')
    env = gym.wrappers.Monitor(env, './monitor', force=True)

    done = True
    progress = tqdm(range(5000))
    for step in progress:
        if done:
            state = env.reset()
        action = 4# env.action_space.sample()
        state, reward, done, info = env.step(action)
        progress.set_postfix(reward=reward)

    env.close()
except KeyboardInterrupt:
    env.close()
