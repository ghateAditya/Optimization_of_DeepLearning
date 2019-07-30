#Importing required libraries and files
import numpy as np
import tensorflow as tf
import argparse
import datetime

from CNN_child import CNN_child
from CNN_Manager import CNN_Manager
from Reinforce import Reinforce

from tensorflow.examples.tutorials.mnist import input_data

#Function to parse command line arguments
def parse_args():
    desc = 'Google Neural Architecture Search Implementation'
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--max_layers',default = 2)
    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args
#End of parse_args function

#Main function
def main():

    #Getting the command line arguments
    global args 
    args = parse_args()

    #Training the RNN
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)

#End of Main function

#Policy Network function to map states and actions
def policy_network(state, max_layers):

    with tf.name_scope('Policy_Network'):

        #Initializing NAS Cell object
        nas_cell = tf.contrib.rnn.NASCell(4*max_layers)

        #Getting the outputs and the state
        outputs, state = tf.nn.dynamic_rnn(nas_cell, tf.expand_dims(state,-1), dtype=tf.float32)
        bias = tf.Variable([0.05]*4*max_layers)

        #Adding the bias
        outputs = tf.nn.bias_add(outputs,bias)
        print("NAS Cell outputs are =====> ", outputs, outputs[:, -1:, :])
        
        return outputs[:,-1:,:]

#End of Policy Network function

#Function to train the Controller RNN
def train(mnist):

    global args
    session = tf.Session()
    global_step = tf.Variable(0, trainable= False, name= 'Global_step')
    learning_rate = tf.train.exponential_decay(0.99,global_step,500,0.96,staircase=True,name='Exponential_Decay')
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    reinforce_obj = Reinforce(session,optimizer,policy_network,args.max_layers,global_step)
    cnn_manager_obj = CNN_Manager(784,10,0.01,mnist)
    
    #Need to tune this
    Max_episodes = 10

    step = 0

    #Initial state
    state = np.array([[10.0,128.0,1.0,1.0]*args.max_layers],np.float32)
    previous_accuracy = 0.0
    total_rewards = 0

    for i_ep in range(Max_episodes):
        action = reinforce_obj.get_action(state)
        print('For Episode: {0}, {1} is the Action'.format(i_ep,action))
        
        if all(i > 0 for i in action[0][0]):
            reward, previous_accuracy = cnn_manager_obj.calculate_reward(action,step,previous_accuracy)
            print('\nReward obtained: {0}, Previous Accuracy: {1}'.format(reward,previous_accuracy))
        else:
            reward = -1.0

        total_rewards += reward
        print('Total rewards accumulated: {0}'.format(total_rewards))

        state = action[0]
        reinforce_obj.storeRollout(state,reward)

        step +=1
        reinforce_loss = reinforce_obj.train_step(1)

        log_str = "Current Time:  "+str(datetime.datetime.now().time())+" Episode:  "+str(i_ep)+" Loss:  "+str(reinforce_loss)+" Last State:  "+str(state)+" Last Reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)

#Main function executor
if __name__ == "__main__":
    main()
#End