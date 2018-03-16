import gym

import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import time
import matplotlib.pyplot as plt




def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 2000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 7  #num of joints being controlled
    state_dim = 14  #num of features in state

    EXPLORE = 200.0*50
    episode_count = 210 if (train_indicator) else 1
    max_steps = 1000 
    reward = 0
    done = False
    step = 0
    epsilon = 0.3 if (train_indicator) else 0.0
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = gym.make('Yumi-Simple-v0')
    env.reset()
    env.render()

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        print("actor done")
        critic.model.load_weights("criticmodel.h5")
        print("critic done")
        actor.target_model.load_weights("actormodel.h5")
        print("actor weights")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    rewardlist=[]
    actionlist0=[]
    actionlist1=[]
    actionlist2=[]
    actionlist3=[]
    actionlist4=[]
    actionlist5=[]
    actionlist6=[]

    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()
        if not train_indicator:
            print("start recording now")
            time.sleep(1)
        s_t = np.array(ob)
     
        total_reward = 0.
        for j in range(max_steps*10):
            env.render()

            loss = 0 
            epsilon -= 0.3 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            if np.random.random() > epsilon:
                a_type = "Exploit"
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))*1 #rescale
            else:
                a_type = "Explore"
                a_t = np.random.uniform(-0.1,0.1, size=(1,7))

            action = np.array([a_t[0,0],a_t[0,1],a_t[0,2],a_t[0,3],a_t[0,4],a_t[0,5],a_t[0,6]])
            ob, r_t, done,info = env.step(action)

            #log
            rewardlist.append(r_t)
            print(a_t.shape)
            actionlist0.append(a_t[0,0])
            actionlist1.append(a_t[0,1])
            actionlist2.append(a_t[0,2])
            actionlist3.append(a_t[0,3])
            actionlist4.append(a_t[0,4])
            actionlist5.append(a_t[0,5])
            actionlist6.append(a_t[0,6])

            s_t1 = np.array(ob)
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_type, "Reward", r_t, "Loss", loss, "Epsilon", epsilon)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                critic.model.save_weights("criticmodel.h5", overwrite=True)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    #env.done()
    print("Finish.")

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


if __name__ == "__main__":
    playGame()
