import gym
import numpy as np
from gym import spaces

#build the enviroment
env = gym.make('Yumi-Simple-v0')
env.reset()
env.render()

#print infos
print("action space: tourg action on the right arm joints [1, 2, 7, 3, 4, 5, 6]")
print(env.action_space)
print("action upper bound")
print(env.action_space.high)
print("action lower bound")
print(env.action_space.low)


print("observationspace:  jointpos [")
print(env.observation_space)
print("observationspace upper bound")
print(env.observation_space.high)
print("observationspace lower bound")
print(env.observation_space.low)

for _ in range(10000): # run for 1000 steps
    env.render()
    #get info from the world
    action = np.zeros(7) #do nothing
    #action = env.action_space.sample() # pick a random action
    print("performing action:")
    print(action)
    observation, reward, done, info = env.step(action)
    print("observing")
    print(observation)
    print("reward")
    print(reward)
    
    env.step(action) # take action