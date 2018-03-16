
import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import tensorflow as tf
import collections

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


print("observationspace:  jointpos [1, 2, 7, 3, 4, 5, 6], jointvel [1, 2, 7, 3, 4, 5, 6], EEpos [x, y, z],EEqats [x y z w] ")
print(env.observation_space)
print("observationspace upper bound")
print(env.observation_space.high)
print("observationspace lower bound")
print(env.observation_space.low)


# class PolicyEstimator():
#     """
#     Policy Function approximator. 
#     """
    
#     def __init__(self, learning_rate=0.01, scope="policy_estimator"):
#         with tf.variable_scope(scope):
#             self.state = tf.placeholder(tf.int32, [], "state")
#             self.action = tf.placeholder(dtype=tf.int32, name="action")
#             self.target = tf.placeholder(dtype=tf.float32, name="target")

#             # This is just table lookup estimator
#             state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
#             self.output_layer = tf.contrib.layers.fully_connected(
#                 inputs=tf.expand_dims(state_one_hot, 0),
#                 num_outputs=env.action_space.n,
#                 activation_fn=None,
#                 weights_initializer=tf.zeros_initializer)

#             self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
#             self.picked_action_prob = tf.gather(self.action_probs, self.action)

#             # Loss and train op
#             self.loss = -tf.log(self.picked_action_prob) * self.target

#             self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#             self.train_op = self.optimizer.minimize(
#                 self.loss, global_step=tf.contrib.framework.get_global_step())
    
#     def predict(self, state, sess=None):
#         sess = sess or tf.get_default_session()
#         return sess.run(self.action_probs, { self.state: state })

#     def update(self, state, target, action, sess=None):
#         sess = sess or tf.get_default_session()
#         feed_dict = { self.state: state, self.target: target, self.action: action  }
#         _, loss = sess.run([self.train_op, self.loss], feed_dict)
#         return loss


# class ValueEstimator():
#     """
#     Value Function approximator. 
#     """
    
#     def __init__(self, learning_rate=0.1, scope="value_estimator"):
#         with tf.variable_scope(scope):
#             self.state = tf.placeholder(tf.int32, [], "state")
#             self.target = tf.placeholder(dtype=tf.float32, name="target")

#             # This is just table lookup estimator
#             state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
#             self.output_layer = tf.contrib.layers.fully_connected(
#                 inputs=tf.expand_dims(state_one_hot, 0),
#                 num_outputs=1,
#                 activation_fn=None,
#                 weights_initializer=tf.zeros_initializer)

#             self.value_estimate = tf.squeeze(self.output_layer)
#             self.loss = tf.squared_difference(self.value_estimate, self.target)

#             self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#             self.train_op = self.optimizer.minimize(
#                 self.loss, global_step=tf.contrib.framework.get_global_step())        
    
#     def predict(self, state, sess=None):
#         sess = sess or tf.get_default_session()
#         return sess.run(self.value_estimate, { self.state: state })

#     def update(self, state, target, sess=None):
#         sess = sess or tf.get_default_session()
#         feed_dict = { self.state: state, self.target: target }
#         _, loss = sess.run([self.train_op, self.loss], feed_dict)
#         return loss


# def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
#     """
#     REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
#     function approximator using policy gradient.
    
#     Args:
#         env: OpenAI environment.
#         estimator_policy: Policy Function to be optimized 
#         estimator_value: Value function approximator, used as a baseline
#         num_episodes: Number of episodes to run for
#         discount_factor: Time-discount factor
    
#     Returns:
#         An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
#     """

#     # Keeps track of useful statistics
#     stats = plotting.EpisodeStats(
#         episode_lengths=np.zeros(num_episodes),
#         episode_rewards=np.zeros(num_episodes))    
    
#     Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
#     for i_episode in range(num_episodes):
#         # Reset the environment and pick the fisrst action
#         state = env.reset()
        
#         episode = []
        
#         # One step in the environment
#         for t in itertools.count():
            
#             # Take a step
#             action_probs = estimator_policy.predict(state)
#             action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
#             next_state, reward, done, _ = env.step(action)
            
#             # Keep track of the transition
#             episode.append(Transition(
#               state=state, action=action, reward=reward, next_state=next_state, done=done))
            
#             # Update statistics
#             stats.episode_rewards[i_episode] += reward
#             stats.episode_lengths[i_episode] = t
            
#             # Print out which step we're on, useful for debugging.
#             print("\rStep {} @ Episode {}/{} ({})".format(
#                     t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
#             # sys.stdout.flush()

#             if done:
#                 break
                
#             state = next_state
    
#         # Go through the episode and make policy updates
#         for t, transition in enumerate(episode):
#             # The return after this timestep
#             total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
#             # Calculate baseline/advantage
#             baseline_value = estimator_value.predict(transition.state)            
#             advantage = total_return - baseline_value
#             # Update our value estimator
#             estimator_value.update(transition.state, total_return)
#             # Update our policy estimator
#             estimator_policy.update(transition.state, advantage, transition.action)
    
#     return stats


# tf.reset_default_graph()

# global_step = tf.Variable(0, name="global_step", trainable=False)
# policy_estimator = PolicyEstimator()
# value_estimator = ValueEstimator()

# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     # Note, due to randomness in the policy the number of episodes you need to learn a good
#     # policy may vary. ~2000-5000 seemed to work well for me.
#     stats = reinforce(env, policy_estimator, value_estimator, 2000, discount_factor=1.0)




rewardlist=[]
actionlist0=[]
actionlist1=[]
actionlist2=[]
actionlist3=[]
actionlist4=[]
actionlist5=[]
actionlist6=[]
foreceslist0=[]
foreceslist1=[]
foreceslist2=[]
foreceslist3=[]
foreceslist4=[]
foreceslist5=[]


for _ in range(1000): # run for 1000 steps
    env.render()
    #get info from the world
    action = np.zeros(7) #do nothing
    #action = env.action_space.sample() # pick a random action
    #print("performing action:")
    #print(action)
    observation, reward, done, info = env.step(action)
    #print("observing")
    #print(observation)
    #print("reward")
    #print(reward)
    rewardlist.append(reward)

    actionlist0.append(action[0])
    actionlist1.append(action[1])
    actionlist2.append(action[2])
    actionlist3.append(action[3])
    actionlist4.append(action[4])
    actionlist5.append(action[5])
    actionlist6.append(action[6])

    foreceslist0.append(observation[-6])
    foreceslist1.append(observation[-5])
    foreceslist2.append(observation[-4])
    foreceslist3.append(observation[-3])
    foreceslist4.append(observation[-2])
    foreceslist5.append(observation[-1])
    #forecvec = np.array([observation[-6], observation[-5], observation[-4], observation[-3], observation[-2], observation[-1]])
    #foreceslist.append(forecvec)

    
    #env.step(action) # take action

#print(len(actionlist[1]))

plt.plot(rewardlist)
plt.title('Reward (distance to goal pose)')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()

#x=np.linspace(0, 1000, num=100, endpoint=True)
plt.plot(actionlist0, label="f_j_r_1")
plt.plot(actionlist1, label="f_j_r_2")
plt.plot(actionlist2, label="f_j_r_7")
plt.plot(actionlist3, label="f_j_r_3")
plt.plot(actionlist4, label="f_j_r_4")
plt.plot(actionlist5, label="f_j_r_5")
plt.plot(actionlist6, label="f_j_r_6")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.axis([0, 1000, -2, 2])

plt.title('Action (joint forces)')
plt.xlabel('episodes')
plt.ylabel('force')
plt.show()


plt.plot(foreceslist0, label="f_x")
plt.plot(foreceslist1, label="f_y")
plt.plot(foreceslist2, label="f_z")
plt.plot(foreceslist3, label="t_x")
plt.plot(foreceslist4, label="t_y")
plt.plot(foreceslist5, label="t_z")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Advisary forces on right hand')
plt.xlabel('episodes')
plt.ylabel('force/torque')
plt.show()
