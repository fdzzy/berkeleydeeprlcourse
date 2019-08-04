import gym
import tensorflow as tf
import numpy as np

def run_gym(env_name, policy_fn, expert_policy_fn=None, num_rollouts=1, render=False, verbose=False):
    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit
    
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        if verbose:
            print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            if expert_policy_fn:
                observations.append(obs)
                actions.append(expert_policy_fn(obs[None,:]))
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0 and verbose: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return observations, actions