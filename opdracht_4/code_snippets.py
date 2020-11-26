conv1 = tf.keras.layers.Conv2D(inputs = input_layer,filters=32,kernel_size=5,padding="same",activation=tf.nn.relu)
pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1 ,pool_size=2,strides=2)
conv2 = tf.keras.layers.Conv2D(inputs=pool1 ,filters=64,kernel_size=5,padding="same",activation=tf.nn.relu)
pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
pool2_flat = tf.reshape(pool2 , [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.keras.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
logits = tf.keras.layers.dense(inputs=dropout, units=10)
tf.argmax(input=logits , axis=1)
tf.nn.softmax(logits , name="softmax_tensor")
predictions = {"classes": tf.argmax(input=logits , axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

def functie1():
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
functie1()

onehot_labels = tf.one_hot(indices=tf.cast(labels , tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

def functie2():
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return
functie2()

tf.estimator.EstimatorSpec(mode=mode , loss=loss , train_op=train_op)

#===========================================================
mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

padding='same', strides=2

'''
import pickle , gzip , os
from urllib import request
from pylab import imshow , show , cm
'''