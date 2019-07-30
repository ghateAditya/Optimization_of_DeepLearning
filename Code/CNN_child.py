import tensorflow as tf

class CNN_child():

    #Constructor to initialize the HyperParameters
    def __init__(self,num_inputs,num_classes,cnn_config):

        #Separating the different parameters from cnn_config
        cnn_Convfilter_size = [c[0] for c in cnn_config]
        cnn_Convfilter_nums = [c[1] for c in cnn_config]
        cnn_MaxPoolfilter_size = [c[2] for c in cnn_config]

        #creating a placeholder for Dropout_rates
        self.cnn_DropoutRates = tf.placeholder(tf.float32, [len(cnn_Convfilter_size),] , name='CNN_dropout_keep_probs')
        self.cnn_Dropout_keep_prob = tf.placeholder(tf.float32, [], name="Dense_dropout_keep_prob")

        #Assigning placeholders for x and y
        self.x = tf.placeholder(tf.float32, shape=[None,num_inputs],name='x')
        self.y = tf.placeholder(tf.int32,[None,num_classes], name='y')

        #Initializing x & y values
        y = self.y
        x = tf.expand_dims(self.x, -1)

        #Setting x as the input layer
        pool_out = x

        with tf.name_scope('Convolutional_Part'):

            #Following loop will be executed twice, can be configured using max layers argument
            for id,filter_size in enumerate(cnn_Convfilter_size):
                with tf.name_scope('L{0}'.format(id)):

                    #Convolutional layer
                    conv_out = tf.layers.conv1d(inputs = pool_out, 
                                                filters = cnn_Convfilter_nums[id],
                                                kernel_size = int(filter_size),
                                                strides= 1,
                                                padding= 'SAME',
                                                name='conv_out{0}'.format(id),
                                                activation= tf.nn.relu,
                                                kernel_initializer= tf.contrib.layers.xavier_initializer(),
                                                bias_initializer= tf.zeros_initializer())

                    #MaxPooling layer
                    pool_out = tf.layers.max_pooling1d(conv_out, pool_size = (int(cnn_MaxPoolfilter_size[id])), strides = 1, padding='SAME',name='maxPool{0}'.format(id))

                    #DropOut layer
                    pool_out = tf.nn.dropout(pool_out, self.cnn_DropoutRates[id])
            #End of loop

            #Flattened layer
            flatten_pred_out = tf.contrib.layers.flatten(pool_out)

            #Final Output layer as per the number of classes
            self.logits = tf.layers.dense(flatten_pred_out, num_classes)
        
        #End of NameScope

        #Applying softmax activation and obtaining predictions
        self.predictions = tf.nn.softmax(self.logits, name='Predictions')

        #Calculating loss
        self.cnn_loss = tf.nn.softmax_cross_entropy_with_logits(logits= self.logits,labels=y,name='CNN_Loss')

        #Calculating model accuracy
        correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(y, 1))
        self.cnn_accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32),name='CNN_Accuracy')

    #End of Constructor

#End of Class





        