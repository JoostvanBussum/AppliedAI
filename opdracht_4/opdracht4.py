import numpy as np
import tensorflow as tf

import pickle , gzip , os
from urllib import request
from pylab import imshow , show , cm

# Virgin reader code
#========================================================================================
"""
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
"""

# Chad tensorflow documented code
#========================================================================================
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

print(predictions)


"""
input_layer = tf.reshape(train_set, [-1, 28, 28, 1])

conv1 = tf.keras.layers.Conv2D(inputs = input_layer,filters=32,kernel_size=5,padding="same",activation='relu')
"""
#=============================================================================================================
