import numpy as np
import tensorflow as tf

import pickle , gzip , os
from urllib import request
from pylab import imshow , show , cm

url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

if not os.path.isfile("mnist.pkl.gz"):
    request.urlretrieve(url, "mnist.pkl.gz")

f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding='latin1')
f.close()

def get_image(number):
    (X, y) = [img[number] for img in train_set]
    return (np.array(X), y)

def view_image(number):
    (X, y) = get_image(number)
    print("Label: %s" % y)
    imshow(X.reshape(28,28), cmap=cm.gray)
    show()


input_layer = tf.reshape(features["x"], [-1, 28, 28,1])
conv1 = tf.layers.conv2d(inputs = input_layer,filters=32,kernel_size=5,padding="same",activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1 ,pool_size=2,strides=2)
conv2 = tf.layers.conv2d(inputs=pool1 ,filters=64,kernel_size=5,padding="same",activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
pool2_flat = tf.reshape(pool2 , [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
logits = tf.layers.dense(inputs=dropout, units=10)
tf.argmax(input=logits , axis=1)
tf.nn.softmax(logits , name="softmax_tensor")
predictions = {"classes": tf.argmax(input=logits , axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

onehot_labels = tf.one_hot(indices=tf.cast(labels , tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return
    
tf.estimator.EstimatorSpec(mode=mode , loss=loss , train_op=train_op)