import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ref
# https://zhuanlan.zhihu.com/p/27865705
# https://zhuanlan.zhihu.com/p/34998569

with open('data.pickle','rb') as f:
    X_train, noisy_X, y_train = pickle.load(f)

learning_rate = 0.0005
training_epochs = 30
batch_size = 400
display_step = 1
latent_dim = 100 # 100
n_input = 784
n_hidden_1 = 500 # 1st layer num features
n_hidden_2 = 250 # 2nd layer num features
weights = dict()
biases = dict()



def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

weights['encoder'] = {
    'encoder_h1': tf.Variable(xavier_init(n_input, n_hidden_1)),
    'encoder_h2': tf.Variable(xavier_init(n_hidden_1, n_hidden_2)),
    'out_mean': tf.Variable(xavier_init(n_hidden_2, latent_dim)),
    'out_log_sigma': tf.Variable(xavier_init(n_hidden_2, latent_dim)),
}
biases['encoder'] = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float32)),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32)),
    'out_mean': tf.Variable(tf.zeros([latent_dim], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([latent_dim], dtype=tf.float32)),
}

weights['decoder'] = {
    'decoder_h1': tf.Variable(xavier_init(latent_dim, n_hidden_1)),
    'decoder_h2': tf.Variable(xavier_init(n_hidden_1, n_hidden_2)),
    'out_mean': tf.Variable(xavier_init(n_hidden_2, n_input)),
    'out_log_sigma': tf.Variable(xavier_init(n_hidden_2, n_input)),
}
biases['decoder'] = {
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float32)),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32)),
    'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
}


def encoder(x):
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder']['encoder_h1']),
                                   biases['encoder']['encoder_b1']))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder']['encoder_h2']),
                                   biases['encoder']['encoder_b2']))

    z_mean = tf.add(tf.matmul(layer_2, weights['encoder']['out_mean']), biases['encoder']['out_mean'])
    z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['encoder']['out_log_sigma']), biases['encoder']['out_log_sigma'])

    return (z_mean, z_log_sigma_sq)


def decoder(x):
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder']['decoder_h1']),
                                   biases['decoder']['decoder_b1']))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['decoder']['decoder_h2']),
                                   biases['decoder']['decoder_b2']))

    x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder']['out_mean']), biases['decoder']['out_mean']))

    return x_reconstr_mean

def bilinear_interpolate(d0, d1, d2, d3, grid=20):
    zn = []
    for i in range(grid):
        for j in range(grid):
            newd = ( i * j * d3 + (grid-1-i) * j * d1 + i * (grid-1-j) * d2 + (grid-1-i) * (grid-1-j) * d0 ) / (grid-1)**2
            zn.append(newd)
    return tf.stack(zn)


X = tf.placeholder(tf.float32, [None, n_input])
z_mean, z_log_sigma_sq = encoder(X)
eps = tf.random_normal((batch_size, latent_dim), 0, 1, dtype=tf.float32)
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

zn = bilinear_interpolate(z[0, :], z[1, :], z[2, :], z[3, :])

# Result from encoder
decoder_op = decoder(z)

# Result from bilinear interpolation
decode_latent = decoder(zn)

# Result from N(0, 1)
sample_output = decoder(eps)

y_pred = decoder_op
y_true = X

regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
reg_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
reg_cost = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

recon_cost = -tf.reduce_sum(y_true * tf.log(1e-10 + y_pred)
                           + (1-y_true) * tf.log(1e-10 + 1 - y_pred), 1)
latent_cost = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(recon_cost + latent_cost + reg_cost)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
cost_hist = []

def next_batch(data1, data2, labels, batch_size):
    idx = np.arange(0, data1.shape[0])
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data1_shuffle = [data1[i] for i in idx]
    data2_shuffle = [data2[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data1_shuffle), np.asarray(data2_shuffle), np.asarray(labels_shuffle)


with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(y_train.shape[0]/batch_size)
    
    for epoch in range(training_epochs):
        batch_cost = []
        for i in range(total_batch):
            batch_xs, batch_noisy, batch_ys = next_batch(X_train, noisy_X, y_train, batch_size)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_xs})
            batch_cost.append(c)
        
        cost_hist.append(-np.mean(batch_cost)/batch_size/784)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))


    plt.clf()
    plt.plot(cost_hist, label='train')
    plt.xlabel('epoch')
    plt.ylabel('lower bound')
    plt.title('Learning curve')
    plt.legend()
    plt.savefig('learning_curve.jpg')


    original, noisy, _ = next_batch(X_train, noisy_X, y_train, batch_size)
    encode_decode, sp, lt = sess.run(
        [y_pred, sample_output, decode_latent], feed_dict={X: original})

    f, a = plt.subplots(3, 4)
    for i in range(4):
        a[0][i].imshow(np.reshape(original[i], (28, 28)).transpose(), cmap='gray')
        a[1][i].imshow(np.reshape(noisy[i], (28, 28)).transpose(), cmap='gray')
        a[2][i].imshow(np.reshape(encode_decode[i], (28, 28)).transpose(), cmap='gray')
    plt.savefig('reconstruct.jpg')


    f, b = plt.subplots(5, 10, figsize=(10, 5))
    for i in range(5):
        for j in range(10):
            b[i][j].imshow(np.reshape(sp[5*i+j], (28, 28)).transpose(), cmap='gray')
    plt.savefig('sample.jpg')

    
    plt.clf()
    plt.figure(figsize=(10,10))
    big_latent = []
    for i in range(20):
        col_image = []
        for j in range(20):
            col_image.append(np.reshape(lt[20*i+j], (28, 28)).transpose())
        big_latent.append(np.hstack(col_image))
    big_latent = np.vstack(big_latent)
    plt.imshow(big_latent, cmap='gray')
    plt.savefig('latent_space.jpg')