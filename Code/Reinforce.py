#Importing the libraries
import tensorflow as tf
import random
import numpy as np

class Reinforce():
    def __init__(self, session, optimizer, policy_network, max_layers, global_step, 
                division_rate = 100.0, reg_param = 0.001, discount_factor = 0.99,
                exploration_prob = 0.25):
        
        #Initializing the variables
        self.session = session
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.max_layers = max_layers
        self.global_step = global_step

        #Initializing two lists to store previous rewards and states
        self.reward_buffer = []
        self.state_buffer = []

        #Calling a function to create all the variables in the graph
        self.create_variables()

        #Running the variables initializer
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))
        
        '''
        Demo Code to be tried out for initialization
        self.session.run(tf.global_variables_initializer())
        '''
    #End of Constructor

    #Function to create all variables in the graph
    def create_variables(self):

        #Defining a placeholder for inputs to the model
        with tf.name_scope('Model_Inputs'):
            self.states = tf.placeholder(tf.float32,[None, self.max_layers*4],name='States')
        #End of Model Inputs name scope

        #Selecting an action based on the previous scope
        with tf.name_scope('Predicting_Actions'):

            #Creating a variable scope for Policy Network which will be shared
            with tf.variable_scope('Policy_Network'):
                self.policy_outputs = self.policy_network(self.states,self.max_layers)
            #End of variable scope

            #Creating a new node based on policy_outputs
            self.action_scores = tf.identity(self.policy_outputs, name='Action_scores')

            #Multiplying with Division rate
            self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate,self.action_scores),tf.int32,name='Predicted_Action')
        #End of name scope

        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Policy_Network')

        #Computing the gradients based on the action taken
        with tf.name_scope('Gradient_Computation'):

            #Shared variable scope
            with tf.variable_scope('Policy_Network', reuse=True):
                self.logprobs = self.policy_network(self.states,self.max_layers)
                #print('Log of Probabilities',self.logprobs)
            #End of variable scope

            #Calculating Policy and Regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs[:,-1,:],labels=self.states)
            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)

            
            self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])

            self.total_loss = self.pg_loss + (self.reg_param*self.reg_loss)

            #Calculating gradients
            self.gradients = self.optimizer.compute_gradients(self.total_loss)

            #Creating a placeholder for Discounted rewards
            self.discounted_rewards = tf.placeholder(tf.float32,(None,), name='Discounted_rewards')
            
            #Multplying by discounted rewards
            for i, (grad,var) in enumerate(self.gradients):
                if grad is not None: self.gradients[i] = (grad*self.discounted_rewards, var)
            
            #Applying the gradients
            with tf.name_scope('Train_Policy_Network'):
                self.reinforce_training_operation = self.optimizer.apply_gradients(self.gradients, global_step = self.global_step)
    
    #End of create_variables function

    #Function to get new action
    def get_action(self,state):
        #if random.random() < self.exploration_prob : return np.array([[random.sample(range(1,25), k= 4*self.max_layers)]])
        #else: return self.session.run(self.predicted_action, {self.states: state})
        return self.session.run(self.predicted_action, {self.states: state})
    #End of get_action function

    #Function to store previous rewards and states
    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])
    #End of storeRollout function

    #Function to execute one reinforcement step and run all the graoh
    def train_step(self, step_count):
        
        #Getting the previous state and reward values
        previous_states = np.array(self.state_buffer[-step_count:])/self.division_rate
        previous_rewards  = np.array(self.reward_buffer[-step_count:])

        #Running the graph and getting the reinforcement loss
        _ , reinforcement_loss = self.session.run([self.reinforce_training_operation, self.total_loss], {self.states: previous_states, self.discounted_rewards: previous_rewards})

        ####### TENSORBOARD STUFF #######
        tf.summary.scalar('reinforcement_loss',reinforcement_loss)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('content/mnist-trial/reinforce/1')
        writer.add_graph(self.session.graph)
        s = self.session.run(merged_summary)
        writer.add_summary(s)
        ################################

        return reinforcement_loss
    #End of train_step function
    
