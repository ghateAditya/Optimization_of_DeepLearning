import tensorflow as tf
from CNN_child import CNN_child

class CNN_Manager():

    #Constructor for initialization of values
    def __init__(self, num_inputs, num_classes, learning_rate, mnist, batch_size = 100, max_steps_per_action = 5500, dropout_rate = 0.85):

        #Initializing all the values
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist
        self.batch_size = batch_size
        self.max_steps_per_action = max_steps_per_action
        self.dropout_rate = dropout_rate
    
    #End of Constructor

    #Function for calculating reward based on accuracy of child CNN
    def calculate_reward(self, action, step, previous_accuracy):
        
        #Converting 3D action list into a list of two lists of CNN states
        action = [action[0][0][x : (x+4)] for x in range(0, len(action[0][0]), 4)]

        #Making a list of CNN dropout rates from action list
        cnn_DropoutRates = [c[3] for c in action]

        #Setting default graph
        with tf.Graph().as_default() as g:
            
            with g.container('Experiment{0}'.format(step)):

                #Creating an instance of the child CNN class
                cnn_model = CNN_child(self.num_inputs, self.num_classes, action)

                #Getting the CNN loss
                cnn_loss_operation = tf.reduce_mean(cnn_model.cnn_loss)

                #Setting up an optimizer to minimize the loss
                cnn_optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
                cnn_training_operation = cnn_optimizer.minimize(cnn_loss_operation)

                #Creating a tensorflow session for training the CNN
                with tf.Session() as cnn_training_session:
                    
                    #Initializing all the variables
                    cnn_training_session.run(tf.global_variables_initializer())

                    #Loop for batchwise training of CNN
                    for step in range(self.max_steps_per_action):

                        #initializing the batch wise x and y
                        batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)

                        #Training the batch
                        feed = {cnn_model.x : batch_x, cnn_model.y : batch_y, cnn_model.cnn_Dropout_keep_prob : self.dropout_rate, cnn_model.cnn_DropoutRates : cnn_DropoutRates}
                        _ = cnn_training_session.run(cnn_training_operation , feed_dict= feed)

                        #Printing loss and accuracy for every 100th iteration
                        if step%100 == 0:
                            batch_feed_dict = {cnn_model.x : batch_x, cnn_model.y : batch_y, cnn_model.cnn_Dropout_keep_prob : 1.0, cnn_model.cnn_DropoutRates : [1.0]*len(cnn_DropoutRates)}
                            cnn_batch_loss, cnn_batch_accuracy = cnn_training_session.run([cnn_loss_operation,cnn_model.cnn_accuracy], feed_dict= batch_feed_dict)
                            print('Step {0}, MiniBatch loss: {1:.4f}, MiniBatch accuracy: {2:.4f}'.format(step,cnn_batch_loss,cnn_batch_accuracy))
                    
                    #End of For loop
 
                    #Test set loss and accuracy
                    batch_x, batch_y = self.mnist.test.next_batch(10000)
                    test_feed_dict = {cnn_model.x : batch_x, cnn_model.y : batch_y, cnn_model.cnn_Dropout_keep_prob : 1.0, cnn_model.cnn_DropoutRates : [1.0]*len(cnn_DropoutRates)}
                    test_loss, test_accuracy = cnn_training_session.run([cnn_loss_operation, cnn_model.cnn_accuracy], feed_dict=test_feed_dict)
                    
                    ####### TENSORBOARD STUFF #######
                    tf.summary.scalar('cnn_loss',test_loss)
                    tf.summary.scalar('cnn_accuracy',test_accuracy)
                    merged_summary = tf.summary.merge_all()
                    writer = tf.summary.FileWriter('content/mnist-trial/cnn/1')
                    writer.add_graph(cnn_training_session.graph)
                    s = cnn_training_session.run(merged_summary)
                    writer.add_summary(s)
                    ################################

                    print('\n-----------\nCurrent Accuracy: {0}, Previous Accuracy: {1}'.format(test_accuracy,previous_accuracy))
                    
                    #Calculating Reward
                    
                    '''
                    Demo code to be tried out 
                    '''
                    if test_accuracy - previous_accuracy <= 0.01 : return -0.01, test_accuracy
                    #if test_accuracy - previous_accuracy <= 0.01 : return test_accuracy, test_accuracy
                    else: return 0.01, test_accuracy