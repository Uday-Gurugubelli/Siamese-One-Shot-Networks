import tensorflow as tf

class SiameseNet:
    def __init__(self, features):
      (self.image1, self.image2) = features
      
    def sub_network(featrues):
        features = tf.reshape(features, [-1, 64, 64, 1])
        
        network = tf.layers.conv2d(inputs=features,
                            filters=64,
                            kernel_size=10,
                            padding='SAME',
                            activation = tf.nn.tanh)
                            #kernel_initializer = tf.contrib.layers.xavier_initializer())
        network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

        network = tf.layers.conv2d(inputs=network,
                            filters=128,
                            kernel_size=7,
                            padding='SAME',
                            activation = tf.nn.elu)
                            #kernel_initializer = tf.contrib.layers.xavier_initializer())
        network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

        network = tf.layers.conv2d(inputs=network,
                         filters=128,
                         kernel_size=4,
                         padding='SAME',
                         activation = tf.nn.relu)
                         #kernel_initializer = tf.contrib.layers.xavier_initializer())
        network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

        network = tf.layers.conv2d(inputs=network,
                         filters=256,
                         kernel_size=4,
                         padding='SAME',
                         activation = tf.nn.relu)
                         #kernel_initializer = tf.contrib.layers.xavier_initializer())
    
        network_flat = tf.reshape(network, [-1, 256*8*8])

        dense_nw = tf.layers.dense(inputs=network_flat, units=4096, activation=tf.nn.sigmoid)
    
        return dense_nw
    
    def final_layer():    
        dense1 = sub_network(self.image1)
        dense2 = sub_network(self.image2)
        l1_dist = tf.reshape(tf.abs(tf.subtract(dense1,dense2)), (BATCH,4096))
    
        y_ = tf.layers.dense(inputs=l1_dist, units=1, activation= tf.nn.sigmoid)
        y_ = tf.reshape(y_, (1,))
        
        return y_
        
