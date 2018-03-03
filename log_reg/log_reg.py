
import numpy as np
import tensorflow as tf

def logistic_regression(loss_func=tf.losses.log_loss):

 

    c1 = np.random.randn(50, 100) + 1
    c2 = np.random.randn(50, 100) - 1
    X = np.vstack([c1, c2])
    Y = np.concatenate([np.ones((50, 1)), np.zeros((50, 1))])
    w = tf.Variable(tf.random_normal((100, 1)),
                    name='w', dtype=tf.float32)
    b = tf.Variable(tf.random_normal((1,)),
                    name='b', dtype=tf.float32)
    input = tf.placeholder(tf.float32, [None, 100])
    pred = tf.nn.sigmoid(tf.matmul(input, w) + b, name='lr_output')
    target = tf.placeholder(tf.float32, [None, 1])
    loss = loss_func(target, pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ls = 1.0
        i = 0
        while ls > 0.05:
            _, ls = sess.run([optimizer, loss], feed_dict={input: X, target: Y})
            if i % 100 == 0:
                print("Loss: {}".format(ls))
            i += 1
        print("Final loss: {}".format(ls))

logistic_regression()
