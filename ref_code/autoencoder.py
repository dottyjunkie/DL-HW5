# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# Visualize decoder setting
# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1
examples_to_show = 10
latent_dim = 20

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
weights = dict()
biases = dict()

weights['encoder'] = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out_mean': tf.Variable(tf.random_normal([n_hidden_2, latent_dim])),
    'out_log_sigma': tf.Variable(tf.random_normal([n_hidden_2, latent_dim])),
}
biases['encoder'] = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out_mean': tf.Variable(tf.zeros([latent_dim], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([latent_dim], dtype=tf.float32)),
}

weights['decoder'] = {
	'decoder_h1': tf.Variable(tf.random_normal([latent_dim, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out_mean': tf.Variable(tf.random_normal([n_hidden_2, n_input])),
    'out_log_sigma': tf.Variable(tf.random_normal([n_hidden_2, n_input])),
}
biases['decoder'] = {
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
}
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder']['encoder_h1']),
    #                                biases['encoder']['encoder_b1']))
    # # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder']['encoder_h2']),
    #                                biases['encoder']['encoder_b2']))
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder']['encoder_h1']),
                                   biases['encoder']['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder']['encoder_h2']),
                                   biases['encoder']['encoder_b2']))

    z_mean = tf.add(tf.matmul(layer_2, weights['encoder']['out_mean']),
                                   biases['encoder']['out_mean'])
    z_log_sigma_sq = \
        tf.add(tf.matmul(layer_2, weights['encoder']['out_log_sigma']),
         biases['encoder']['out_log_sigma'])

    return (z_mean, z_log_sigma_sq)


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder']['decoder_h1']),
    #                                biases['decoder']['decoder_b1']))
    # # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder']['decoder_h2']),
    #                                biases['decoder']['decoder_b2']))
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder']['decoder_h1']),
                                   biases['decoder']['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['decoder']['decoder_h2']),
                                   biases['decoder']['decoder_b2']))
    # z_mean = tf.add(tf.matmul(layer_2, weights['decoder']['out_mean']),
    #                                biases['decoder']['out_mean'])
    # z_log_sigma_sq = \
    #     tf.add(tf.matmul(layer_2, weights['decoder']['out_log_sigma']),
    #      biases['decoder']['out_log_sigma'])
    x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder']['out_mean']),
     biases['decoder']['out_mean']))

    return x_reconstr_mean


"""

# Visualize encoder setting
# Parameters
learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 10
batch_size = 256
display_step = 1

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4
"""

# Construct model
# encoder_op = encoder(X)
z_mean, z_log_sigma_sq = encoder(X)
eps = tf.random_normal((batch_size, latent_dim))
z = tf.add(z_mean, tf.multiply(
	tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

# decoder_op = decoder(encoder_op)
decoder_op = decoder(z)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# recon_cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
recon_cost = -tf.reduce_sum(y_true * tf.log(1e-10 + y_pred)
                           + (1-y_true) * tf.log(1e-10 + 1 - y_pred), 1)
latent_cost = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), axis=1)
# latent_cost = 0
cost = tf.reduce_mean(recon_cost + latent_cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:batch_size]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    # print(encode_decode)
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()
