"""
    Find a minimum of
    f(x) := (Ax - b) ^2
    (A not necessarily invertible)
    with gradient descent
"""
import numpy as np
import tensorflow as tf

x = tf.Variable(tf.random_normal((100, 1)), name="x", dtype=tf.float32)
A = tf.constant(value=np.random.normal(size=(20, 100)), dtype=tf.float32)
b = tf.constant(value=np.random.normal(size=(20,)), dtype=tf.float32)
f = 0.5 * tf.reduce_sum(tf.square(tf.matmul(A, x) - b))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(f)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    current_value = 0.0
    new_value = 1.0
    i = 0
    while np.abs(current_value - new_value) / new_value > 1e-6:
        i += 1
        current_value = new_value
        _, new_value, x_value = sess.run([optimizer, f, x], feed_dict={})
        if i % 1000 == 0:
            print("current minimum: {}".format(new_value))
    print("final minimum: {}".format(new_value))
