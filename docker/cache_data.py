import tensorflow_datasets as tfds
import tensorflow as tf

def cache_mnist():
    # assert 'mnist' in tfds.list_builders()
    # tfds.load('mnist', split='train', shuffle_files=True)
    tf.keras.datasets.mnist.load_data()
    tf.keras.datasets.fashion_mnist.load_data()
    tf.keras.datasets.cifar10.load_data()


if __name__ == '__main__':
    cache_mnist()
