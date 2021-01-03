import tensorflow as tf


def cache_datasets():
    tf.keras.datasets.mnist.load_data()
    tf.keras.datasets.fashion_mnist.load_data()
    tf.keras.datasets.cifar10.load_data()


if __name__ == '__main__':
    cache_datasets()
