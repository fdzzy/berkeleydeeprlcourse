import gym
import time

#env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
#env = gym.make('MsPacman-v0')
env = gym.make('Hopper-v2')
#env.reset()

print("action_space: {}".format(env.action_space))

for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action) # take a random action
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        time.sleep(0.01)
env.close()
