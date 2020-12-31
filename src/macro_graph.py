import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import os
import pickle


DATA_DIR = '../data/mnist/'

def load_mnist(as_np=False):

    tf_filename = 'tf_mnist_shfl_train.tfds'
    np_filename = 'np_mnist_shfl_train.pkl'
    if not os.path.exists(DATA_DIR + np_filename) or not os.path.exists(DATA_DIR + tf_filename):
        assert 'mnist' in tfds.list_builders()
        mnist = tfds.load('mnist', split='train', shuffle_files=True)
        tf.data.experimental.save(mnist, DATA_DIR + tf_filename)
        assert isinstance(mnist, tf.data.Dataset)
        mnist_np = np.array([(ex['image'], ex['label']) for ex in tfds.as_numpy(mnist)])
        with open(DATA_DIR + np_filename, 'wb') as f:
            pickle.dump(mnist_np, f)

    if as_np:
        with open(DATA_DIR + np_filename, 'rb') as f:
            mnist_np = pickle.load(f)

        samples = np.array([s for s in mnist_np[:, 0]], dtype=np.uint8)
        labels = np.array(mnist_np[:, 1], dtype=np.uint8)
        return samples, labels
    else:
        mnist_tf = tf.data.experimental.load(DATA_DIR + tf_filename, tf.TensorSpec(shape=(), dtype=tf.int64))
        return mnist_tf

class GlobalAveragePool(tf.keras.layers.Layer):
    def __init__(self, out_dim, height_dim=1, width_dim=2):
        super(GlobalAveragePool, self).__init__()
        self.out_dim = out_dim
        self.height_dim = height_dim
        self.width_dim = width_dim



    def build(self, input_shape):
        pass

    def call(self, input_tensors, **kwargs):
        return tf.reduce_mean(input_tensors, axis=[self.height_dim, self.width_dim])


class GNASConcat(tf.keras.layers.Layer):
    def __init__(self, pool_layer: tf.keras.layers.Layer, out_dim):
        super(GNASConcat, self).__init__()
        self.pool_layer = pool_layer
        self.pool_size = pool_layer.pool_size
        self.out_dim = out_dim

    def build(self, input_shape):
        pass

    def poolable(self, start_dim, target_dim, pool_size):
        if self.pool_layer.padding != 'valid':
            raise NotImplementedError("Can only use GNASConcat with pooling padding = 'valid'.")
        while target_dim < start_dim:
            start_dim //= pool_size
        return start_dim == target_dim

    def call(self, input_tensors, **kwargs):
        height_dim = 1
        width_dim = 2
        shapes = np.array([np.asarray(it.shape) for it in input_tensors])
        smallest_height = np.min(shapes[:, height_dim])
        smallest_width = np.min(shapes[:, width_dim])
        for shape in shapes:
            assert self.poolable(shape[height_dim], smallest_height, self.pool_size[0]), \
                f"Inputs with heights {shape[height_dim]}, {smallest_height} cannot be consolidated using pool height = " \
                f"{self.pool_size[0]}"
            assert self.poolable(shape[width_dim], smallest_width, self.pool_size[1]), \
                f"Inputs with widths {shape[width_dim]}, {smallest_width} cannot be consolidated using pool width = " \
                f"{self.pool_size[1]}"

        pooled_inputs = []
        for input_tensor in input_tensors:
            while input_tensor.shape[height_dim] > smallest_height:
                input_tensor = self.pool_layer(input_tensor)
            assert input_tensor.shape[height_dim] == smallest_height and input_tensor.shape[width_dim] == smallest_width, \
                "Error while pooling input tensors for GNASConcat."
            pooled_inputs.append(input_tensor)

        output = tf.concat(pooled_inputs, axis=3)
        pointwise_conv = tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=(1, 1))
        return pointwise_conv(output)


def build_macro_graph(input_channels):

    op_types = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
    n_layers = 6
    out_channels = [8, 16, 32, 64, 128, 256]
    in_channels = [input_channels]
    for i in range(len(out_channels) - 1):
        in_channels.append(out_channels[i])
    depthwise_rounding = 'up'

    graph_layers = []

    def get_depthwise_multiplier(layer_id, rounding):
        if rounding == 'up':
            return math.ceil(out_channels[layer_id] / in_channels[layer_id])
        elif rounding == 'down':
            return math.floor(out_channels[layer_id] / in_channels[layer_id])
        else:
            raise ValueError("Depthwise rounding must be either 'down' or 'up'.")

    for L in range(n_layers):
        layer = []

        for op_type in op_types:
            layer_ops = {}
            layer_ops_id = f'{op_type}_L{L}'
            layer_ops['id'] = layer_ops_id
            if op_type == 'conv3x3':
                ops = [GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                  out_dim=out_channels[L]),
                       tf.keras.layers.Conv2D(filters=out_channels[L], kernel_size=(3, 3), strides=(1, 1),
                                              padding='same', kernel_initializer='glorot_uniform'),
                       tf.keras.layers.Activation(activation=tf.keras.activations.relu)]
            elif op_type == 'conv5x5':
                ops = [GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                  out_dim=out_channels[L]),
                       tf.keras.layers.Conv2D(filters=out_channels[L], kernel_size=(5, 5), strides=(1, 1),
                                              padding='same', kernel_initializer='glorot_uniform')]
            elif op_type == 'sep3x3':
                depth_multiplier = get_depthwise_multiplier(L, depthwise_rounding)
                ops = [GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                  out_dim=out_channels[L]),
                       tf.keras.layers.SeparableConv2D(filters=out_channels[L], kernel_size=(3, 3),
                                                       depth_multiplier=depth_multiplier, padding='same',
                                                       depthwise_initializer='glorot_uniform',
                                                       pointwise_initializer='glorot_uniform')]
            elif op_type == 'sep5x5':
                depth_multiplier = get_depthwise_multiplier(L, depthwise_rounding)
                ops = [GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                  out_dim=out_channels[L]),
                       tf.keras.layers.SeparableConv2D(filters=out_channels[L], kernel_size=(5, 5),
                                                       depth_multiplier=depth_multiplier, padding='same',
                                                       depthwise_initializer='glorot_uniform',
                                                       pointwise_initializer='glorot_uniform')]
            elif op_type == 'maxpool':
                ops = [GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                  out_dim=out_channels[L]),
                       tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')]
            elif op_type == 'avgpool':
                ops = [GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                  out_dim=out_channels[L]),
                       tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')]
            else:
                raise NotImplementedError(f"Found op type '{op_type}' but no corresponding implementation.")

            layer_ops['ops'] = ops
            layer.append(layer_ops)

        graph_layers.append(layer)

    containing_components = {
        'input_layer': tf.keras.layers.Input(shape=(28, 28, 1)),
        'final_concat': GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
                                   out_dim=256),
        'global_avg_pool': GlobalAveragePool(out_dim=256),
        'fc': tf.keras.layers.Dense(units=10, input_shape=(256,)),
        'softmax': tf.keras.layers.Softmax()
    }

    return graph_layers, containing_components

def build_from_actions(macro_graph, containing_components, op_actions, connections):
    input_layer = containing_components['input_layer']

    subgraph_layers = get_subgraph_layers(macro_graph, input_layer, op_actions, connections)

    fc_input = containing_components['final_concat'](subgraph_layers)
    fc_input = containing_components['global_avg_pool'](fc_input)
    out = containing_components['fc'](fc_input)
    out = containing_components['softmax'](out)

    model = tf.keras.Model(inputs=input_layer, outputs=out)
    # print(model.summary())
    # tf.keras.utils.plot_model(model, 'testmodel.png')
    return model


def get_subgraph_layers(macro_graph, input_layer, op_actions, connections):
    custom_layers = [macro_graph[i][op_actions[i]] for i in range(len(op_actions))]

    layer_outputs = [input_layer]
    dangling = [True] * len(custom_layers)  # first is for input; last layer will always dangle

    for layer_ops, connections in zip(custom_layers, connections):
        layer_inputs = []
        for i in range(len(connections)):
            if connections[i] == 1 and layer_outputs[i] is not None:
                dangling[i] = False
                layer_inputs.append(layer_outputs[i])
        if not layer_inputs:
            layer_outputs.append(None)
            continue
        out = layer_inputs
        for op in layer_ops['ops']:
            out = op(out)
        layer_outputs.append(out)

    fc_input = []
    for i in range(len(dangling)):
        if dangling[i] and layer_outputs[i] is not None:
            fc_input.append(layer_outputs[i])
    if layer_outputs[-1] is not None:  # last layer always dangling but may be None if no inputs
        fc_input.append(layer_outputs[-1])

    return fc_input


def train_model():
    op_actions = [0, 4, 3, 2, 5, 5]
    connections = [[1], [0, 1], [0, 0, 0], [1, 0, 1, 1], [0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
    macro_graph, containing_components = build_macro_graph(input_channels=1)
    model = build_from_actions(macro_graph, containing_components, op_actions, connections)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    images, labels = load_mnist(as_np=True)

    permute = np.random.permutation(len(images))
    images, labels = images[permute], labels[permute]

    with tf.GradientTape() as tape:
        result = model(images[:128])
        loss = loss_fn(labels[:128], result)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(model.trainable_variables[0][0][0][0][0])


    op_actions = [1, 4, 1, 2, 3, 5]
    connections = [[1], [0, 1], [0, 0, 0], [1, 0, 1, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]]

    model2 = build_from_actions(macro_graph, containing_components, op_actions, connections)
    print(model2.trainable_variables[0][0][0][0][0])
    with tf.GradientTape() as tape:
        result = model2(images[128:256])
        loss = loss_fn(labels[128:256], result)
    gradients = tape.gradient(loss, model2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))

    print(model.trainable_variables[0][0][0][0][0])
    print(model2.trainable_variables[0][0][0][0][0])


    print('test')




if __name__ == '__main__':

    # graph_layers = build_macro_graph(input_shape=(128, 28, 28, 1))
    train_model()
    print('test')


