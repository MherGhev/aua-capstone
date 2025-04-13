import tensorflow as tf

with tf.device('/GPU:0'):
    print("hello world")    