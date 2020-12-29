import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import pickle

DATA_DIR = '../data/mnist/'

def load_mnist():

    filename = 'mnist_shfl_train.pkl'
    if not os.path.exists(DATA_DIR + filename):
        assert 'mnist' in tfds.list_builders()
        mnist = tfds.load('mnist', split='train', shuffle_files=True)
        assert isinstance(mnist, tf.data.Dataset)
        mnist_np = np.array([(ex['image'], ex['label']) for ex in tfds.as_numpy(mnist)])
        with open(DATA_DIR + filename, 'wb') as f:
            pickle.dump(mnist_np, f)

    with open(DATA_DIR + filename, 'rb') as f:
        mnist_np = pickle.load(f)

    print('test')
    samples = np.array([s for s in mnist_np[:, 0]], dtype=np.uint8)
    labels = np.array(mnist_np[:, 1], dtype=np.uint8)
    return samples, labels


def build_model():
    x = tf.ones(shape=(1, 28, 28, 1))
    print(x.shape)
    conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last')
    x = conv(x)
    print(f'x shape = {x.shape}')
    pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')
    x1 = pool(x)
    print(f'x1 shape = {x1.shape}')
    conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), padding='same', data_format='channels_last')
    x2 = conv2(x)
    print(f'x2 shape = {x2.shape}')
    x3 = tf.concat([x, x2], axis=3)
    print(f'x3 shape = {x3.shape}')
    conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', data_format='channels_last')
    print(f'x3 shape = {x3.shape}')
    upsample = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')
    x4 = upsample(x1)
    print(f'x4 shape = {x4.shape}')
    x5 = tf.concat([x, x4], axis=3)
    print(f'x5 shape = {x5.shape}')
    transposed_conv = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(2, 2), strides=(2, 2), padding='valid')
    x6 = transposed_conv(x1)
    print(f'x6 shape = {x6.shape}')
    x7 = tf.concat([x, x6], axis=3)
    print(f'x7 shape = {x7.shape}')
    x8 = conv3(x7)
    print(f'x8 shape = {x8.shape}')
    print('test')

if __name__ == '__main__':
    build_model()
    mnist = load_mnist()
