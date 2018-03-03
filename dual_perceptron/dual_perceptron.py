import numpy as np
import tensorflow as tf

def dual_perceptron():



    c1 = np.random.randn(50, 75) + 1
    c2 = np.random.randn(50, 75) - 1
    X = np.vstack([c1, c2])
    Y = np.concatenate([np.ones((50, 1)), 1 - np.zeros((50, 1))])
    alpha = tf.Variable(tf.random_normal((100, 1)),
                    name='w', dtype=tf.float32)
    input = tf.placeholder(tf.float32, [None, 75])
    target = tf.placeholder(tf.float32, [None, 1])

    x_dp = tf.matmul(input, tf.transpose(input))
    yh = tf.reduce_sum(tf.matmul(tf.transpose(alpha * target), x_dp), axis=0)

    loss = 1.0 / 100.0 * tf.reduce_sum(tf.maximum(0.0, yh))
    optimizer = tf.train.AdamOptimizer(learning_rate=5e-3).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ls = 1.0
        i = 0
        while ls > 0.05:
            _, ls = sess.run([optimizer, loss], feed_dict={input: X, target: Y})
            if i % 1000 == 0:
                print("Loss: {}".format(ls))
            i += 1
        print("Final loss: {}".format(ls))

dual_perceptron()
