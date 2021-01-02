import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_transform as tft
import numpy as np
import math
import os
import pickle
import time
import matplotlib.pyplot as plt
from functools import partial


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
        mnist = tfds.load('mnist', split='train', shuffle_files=True)
        mnist_tf = tf.data.experimental.load(DATA_DIR + tf_filename, tf.TensorSpec(shape=(), dtype=tf.int64))
        return mnist_tf

class GlobalAveragePool(tf.keras.layers.Layer):
    def __init__(self, out_dim, height_dim=1, width_dim=2):
        super(GlobalAveragePool, self).__init__()
        self.out_dim = out_dim
        self.height_dim = height_dim
        self.width_dim = width_dim



    # def build(self, input_shape):
    #     pass

    def call(self, input_tensors, **kwargs):
        return tf.reduce_mean(input_tensors, axis=[self.height_dim, self.width_dim])


class GNASPool(tf.keras.layers.Layer):
    def __init__(self, pool_layer: tf.keras.layers.Layer, out_dim):
        super(GNASPool, self).__init__()
        self.pool_layer = pool_layer
        self.out_dim = out_dim
        self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=(1, 1))

    def build(self, input_shape):
        input_shape_arr = np.array(input_shape)
        input_shape_arr[1] = input_shape[1] // self.pool_layer.pool_size[0]
        input_shape_arr[2] = input_shape[2] // self.pool_layer.pool_size[1]
        self.pointwise_conv.build(input_shape=input_shape_arr)

    def call(self, input_tensor, **kwargs):
        out = self.pool_layer(input_tensor)
        out = self.pointwise_conv(out)
        return out


class GNASConcat(tf.keras.layers.Layer):
    def __init__(self, pool_layer: tf.keras.layers.Layer, out_dim):
        super(GNASConcat, self).__init__()
        self.pool_layer = pool_layer
        self.pool_size = pool_layer.pool_size
        self.out_dim = out_dim
        # self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=(1, 1))
        # self.batchnorm = tf.keras.layers.BatchNormalization()
        # self.relu = tf.keras.layers.Activation(tf.keras.activations.relu)

    # def build(self, input_shape):
    #     # print(f'built GNASConcat with {self.out_dim} out channels in input shape = {input_shape}')
    #     # self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=(1, 1))
    #     pass

    def poolable(self, start_dim, target_dim, pool_size):
        if self.pool_layer.padding != 'valid':
            raise NotImplementedError("Can only use GNASConcat with pooling padding = 'valid'.")
        while target_dim < start_dim:
            start_dim //= pool_size
        return start_dim == target_dim

    def handle_multi_input(self, input_tensors):
        height_dim = 1
        width_dim = 2
        channel_dim = 3
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
            assert input_tensor.shape[height_dim] == smallest_height and input_tensor.shape[
                width_dim] == smallest_width, \
                "Error while pooling input tensors for GNASConcat."
            pooled_inputs.append(input_tensor)

        output = tf.concat(pooled_inputs, axis=channel_dim)

        # debugging purposes
        # self.batchnorm = tf.keras.layers.BatchNormalization()
        # self.relu = tf.keras.layers.Activation(tf.keras.activations.relu)
        # self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.out_dim, kernel_size=(1, 1))
        # output = self.batchnorm(output)

        # print(output.shape)

        # output = self.pointwise_conv(output)
        # output = self.relu(output)
        return output

    # def return_input_tensors(self, input_tensors):
    #     return input_tensors
    # def return_nonzero_input_tensor(self, input_tensors, index):
    #     return input_tensors[index]

    #  TODO: only concatenate if multiple non-zero input tensors present
    def call(self, input_tensors, **kwargs):
        # nonzeros = [tf.cond(tf.math.count_nonzero(i_tensor) > 0, 1, 0) for i_tensor in input_tensors]
        # nonzeros = [int(tf.math.count_nonzero(i_tensor) > 0) for i_tensor in input_tensors]
        # if sum(nonzeros) == 1:
        #     return input_tensors[nonzeros.index(1)]

        # nonzeros = [tf.cast(tf.greater(tf.math.count_nonzero(it), 0), tf.int32) for it in input_tensors]
        # n_nonzeros = tf.reduce_sum(nonzeros)
        # return tf.cond(tf.equal(sum(nonzeros), 1), input_tensors[nonzeros.index(1)], input_tensors)
        # def true_fn: return input_tensors[tf]

        return self.handle_multi_input(input_tensors)

def get_conv_ops(conv_op, layer, in_channels, out_channels, regulariser_fn=None):
    return [
        GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                   out_dim=in_channels[layer]),
        tf.keras.layers.Conv2D(filters=in_channels[layer], kernel_size=(1, 1), kernel_regularizer=regulariser_fn,
                               bias_regularizer=regulariser_fn),
        tf.keras.layers.Activation(tf.keras.activations.relu),

        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Activation(activation=tf.keras.activations.relu),
        conv_op,
        # tf.keras.layers.BatchNormalization()
        tf.keras.layers.Activation(activation=tf.keras.activations.relu)
        # conv_op,
    ]

def get_pool_ops(pool_op, layer, in_channels, out_channels, regulariser_fn=None):
    return [
        GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'), out_dim=out_channels[layer]),
        # tf.keras.layers.Conv2D(filters=out_channels[layer], kernel_size=(1, 1)),
        # tf.keras.layers.Layer(),  # so skip the same number of layers as conv ops
        tf.keras.layers.Conv2D(filters=in_channels[layer], kernel_size=(1, 1), kernel_regularizer=regulariser_fn,
                               bias_regularizer=regulariser_fn),
        tf.keras.layers.Activation(tf.keras.activations.relu),
        # tf.keras.layers.Conv2D(filters=out_channels[layer], kernel_size=(1, 1)),
        pool_op,
    ]


def build_macro_graph(input_shape, out_channels, regularizer=None, regularization=1e-5):

    op_types = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
    in_channels = [input_shape[-1]]
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

    def get_regularizer(regulariser, rate):
        if regulariser is None:
            return None
        elif regulariser == 'l2':
            return tf.keras.regularizers.L2(l2=rate)
        elif regulariser == 'l1':
            return tf.keras.regularizers.L1(l1=rate)
        else:
            raise NotImplementedError("Specify either 'l1' or 'l2' regularization.")

    for L in range(len(out_channels)):
        layer = []

        for op_type in op_types:
            layer_ops = {}
            layer_ops_id = f'{op_type}_L{L}'
            layer_ops['id'] = layer_ops_id
            if op_type == 'conv3x3':
                op = tf.keras.layers.Conv2D(filters=out_channels[L], kernel_size=(3, 3), padding='same',
                                            kernel_regularizer=get_regularizer(regularizer, regularization),
                                            bias_regularizer=get_regularizer(regularizer, regularization))
                ops = get_conv_ops(op, L, in_channels, out_channels,
                                   regulariser_fn=partial(get_regularizer(regularizer, regularization)))
            elif op_type == 'conv5x5':
                op = tf.keras.layers.Conv2D(filters=out_channels[L], kernel_size=(5, 5), padding='same',
                                            kernel_regularizer=get_regularizer(regularizer, regularization),
                                            bias_regularizer=get_regularizer(regularizer, regularization))
                ops = get_conv_ops(op, L, in_channels, out_channels,
                                   regulariser_fn=partial(get_regularizer(regularizer, regularization)))
            elif op_type == 'sep3x3':
                depth_multiplier = get_depthwise_multiplier(L, depthwise_rounding)
                op = tf.keras.layers.SeparableConv2D(filters=out_channels[L], kernel_size=(3, 3),
                                                     depth_multiplier=depth_multiplier, padding='same',
                                                     kernel_regularizer=get_regularizer(regularizer, regularization),
                                                     bias_regularizer=get_regularizer(regularizer, regularization))
                ops = get_conv_ops(op, L, in_channels, out_channels,
                                   regulariser_fn=partial(get_regularizer(regularizer, regularization)))
            elif op_type == 'sep5x5':
                depth_multiplier = get_depthwise_multiplier(L, depthwise_rounding)
                op = tf.keras.layers.SeparableConv2D(filters=out_channels[L], kernel_size=(5, 5),
                                                     depth_multiplier=depth_multiplier, padding='same',
                                                     kernel_regularizer=get_regularizer(regularizer, regularization),
                                                     bias_regularizer=get_regularizer(regularizer, regularization))
                ops = get_conv_ops(op, L, in_channels, out_channels,
                                   regulariser_fn=partial(get_regularizer(regularizer, regularization)))
            elif op_type == 'maxpool':
                #  to keep channels consistent, use the OUT channels for concat, not IN,
                #  as pooling layers don't alter depth
                op = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')
                op = GNASPool(op, out_dim=out_channels[L])
                ops = get_pool_ops(op, L, in_channels, out_channels,
                                   regulariser_fn=partial(get_regularizer(regularizer, regularization)))
            elif op_type == 'avgpool':
                op = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')
                ops = get_pool_ops(op, L, in_channels, out_channels,
                                   regulariser_fn=partial(get_regularizer(regularizer, regularization)))
            else:
                raise NotImplementedError(f"Found op type '{op_type}' but no corresponding implementation.")

            layer_ops['ops'] = ops
            layer.append(layer_ops)

        graph_layers.append(layer)

    containing_components = {
        'input_layer': tf.keras.layers.Input(shape=input_shape[1:]),
        'final_concat': GNASConcat(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                                   out_dim=out_channels[-1]),
        'global_avg_pool': GlobalAveragePool(out_dim=out_channels[-1]),
        # 'final_batchnorm': tf.keras.layers.BatchNormalization(),
        'flatten': tf.keras.layers.Flatten(),
        'fc': tf.keras.layers.Dense(units=10, input_shape=(out_channels[-1],),
                                    kernel_regularizer=get_regularizer(regularizer, regularization),
                                    bias_regularizer=get_regularizer(regularizer, regularization)),
        'softmax': tf.keras.layers.Softmax()
    }

    return graph_layers, containing_components

def build_from_actions(macro_graph, containing_components, op_actions, connections):
    input_layer = containing_components['input_layer']

    subgraph_layers = get_subgraph_layers(macro_graph, input_layer, op_actions, connections)

    # fc_input = containing_components['final_concat'](subgraph_layers)
    # fc_input = containing_components['global_avg_pool'](fc_input)
    fc_input = containing_components['global_avg_pool'](subgraph_layers[0])
    # fc_input = containing_components['flatten'](subgraph_layers[0])
    # fc_input = containing_components['flatten'](fc_input)
    # fc_input = containing_components['final_batchnorm'](fc_input)
    out = containing_components['fc'](fc_input)
    out = containing_components['softmax'](out)

    model = tf.keras.Model(inputs=input_layer, outputs=out)
    # print(model.summary())
    # tf.keras.utils.plot_model(model, 'testmodel.png')
    # print(model.summary())
    return model


def get_subgraph_layers(macro_graph, input_layer, op_actions, connections):
    custom_layers = [macro_graph[i][op_actions[i]] for i in range(len(op_actions))]
    layer_outputs = [input_layer]

    n_multitensor_ops = 3
    n_multitensor_skips = 0

    for layer_ops, connections in zip(custom_layers, connections):
        layer_inputs = []
        nonzero_inputs = 0
        for i in range(len(connections)):
            if connections[i] == 1:
                layer_inputs.append(layer_outputs[i])
                nonzero_inputs += 1
                nonzero_index = i
            else:
                layer_inputs.append(tf.zeros_like(layer_outputs[i]))
        out = layer_inputs
        for op in layer_ops['ops']:
            if n_multitensor_skips % n_multitensor_ops != 0:
                n_multitensor_skips += 1
                continue
            if nonzero_inputs == 1 and isinstance(op, GNASConcat):
                n_multitensor_skips += 1
                out = layer_inputs[nonzero_index]
                continue
            out = op(out)
        layer_outputs.append(out)

    return [out]


def get_subgraph_layers_old(macro_graph, input_layer, op_actions, connections):
    custom_layers = [macro_graph[i][op_actions[i]] for i in range(len(op_actions))]

    layer_outputs = [input_layer]
    dangling = [True] * len(custom_layers)  # first is for input; last layer will always dangle

    for layer_ops, connections in zip(custom_layers, connections):
        layer_inputs = []
        for i in range(len(connections)):
            if connections[i] == 1 and layer_outputs[i] is not None:
                dangling[i] = False
                layer_inputs.append(layer_outputs[i])
            elif sum(connections) > 1:  # i.e. if there are skip connections somewhere, but this isn't one
                layer_inputs.append(tf.zeros_like(layer_outputs[i]))

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


def prepare_data(dataset='mnist', batch_size=128, shuffle=True):

    if dataset == 'mnist':
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError("Dataset must be in ['mnist', 'fashion_mnist', 'cifar10'].")



    mu = np.mean(xtrain)
    sigma = np.std(xtrain)
    xtrain = (xtrain - mu) / sigma
    # xtrain = xtrain / 255.
    xtest = (xtest - mu) / sigma
    if dataset in ['mnist', 'fashion_mnist']:
        xtrain = xtrain[:, :, :, np.newaxis]
        xtest = xtest[:, :, :, np.newaxis]

    if shuffle:
        seed = 0
        # train_permute = np.random.RandomState(seed=seed).permutation(len(xtrain))
        np.random.RandomState(seed=seed).shuffle(xtrain)
        np.random.RandomState(seed=seed).shuffle(ytrain)
        # xtrain2, ytrain2 = xtrain[train_permute], ytrain[train_permute]

    num_train_batches = len(xtrain) // batch_size
    remainder_train = len(xtrain) % batch_size
    xtrain_batches = np.split(xtrain[:-remainder_train], num_train_batches, axis=0)
    ytrain_batches = np.split(ytrain[:-remainder_train], num_train_batches)

    num_test_batches = len(xtest) // batch_size
    remainder_test = len(xtest) % batch_size
    xtest_batches = np.split(xtest[:-remainder_test], num_test_batches, axis=0)
    ytest_batches = np.split(ytest[:-remainder_test], num_test_batches)

    return (xtrain_batches, ytrain_batches), (xtest_batches, ytest_batches)


def generate_random_architecture(num_layers=6):
    # op_actions = np.random.choice(6, num_layers)
    op_actions = np.random.choice(5, num_layers - 1)
    op_actions = np.concatenate([op_actions, np.random.choice(4, 1)])
    connections = [np.random.choice(2, i) for i in range(1, num_layers+1)]
    for connection in connections:
        connection[-1] = 1
    return op_actions, connections

def validate_architecture(op_actions, connections, input_shape, pool_size):
    pool_ops = len([op for op in op_actions if op == 4 or op == 5])
    input_shape = np.array(input_shape)
    for i in range(pool_ops):
        input_shape[1] //= pool_size[0]
        input_shape[2] //= pool_size[1]
    return input_shape[1] > 0 and input_shape[2] > 0


def save_macro(macro_graph, container, filename, template=None):
    if template is not None and os.path.exists(template):
        with open(template, 'rb') as f:
            template_pkl, template_container_shape, template_container_weights = pickle.load(f)
    else:
        template_pkl, template_container_shape, template_container_weights = None, None, None
    macro_pkl = [None] * len(macro_graph)
    for i in range(len(macro_graph)):
        layer = macro_graph[i]
        layer_weights = [None] * len(layer)
        for j in range(len(layer)):
            ops_node = layer[j]
            ops = ops_node['ops']
            ops_weights = [None] * len(ops)
            for k in range(len(ops)):
                if ops[k].weights:
                    if i == 0 and j == 0 and k == 3:
                        t = 4
                    ops_k_weights = {}
                    ops_k_weights['weights'] = ops[k].get_weights()
                    if template_pkl is not None:
                        ops_k_weights['shape'] = template_pkl[i][j][k]['shape']
                    else:
                        ops_k_weights['shape'] = ops[k].input.shape
                    # for weight in ops[k].get_weights():
                    #     ops_k_weights.append(weight)
                    # ops_weights[k] = ops[k].get_weights()
                    ops_weights[k] = ops_k_weights
            layer_weights[j] = ops_weights
        macro_pkl[i] = layer_weights

    if template_container_shape is not None:
        container_fc_shape = template_container_shape
    else:
        container_fc_shape = container['fc'].input.shape
    container_fc_weights = container['fc'].get_weights()
    # t = 4
    with open(filename, 'wb') as f:
        pickle.dump((macro_pkl, container_fc_shape, container_fc_weights), f)
    # with open(filename, 'rb') as f:
    #     test = pickle.load(f)
    # t = 5


def restore_macro(filename, input_shape, out_channels):
    new_macro, new_container = build_macro_graph(input_shape, out_channels)
    with open(filename, 'rb') as f:
        macro_pkl, container_fc_shape, container_fc_weights = pickle.load(f)
    for i in range(len(new_macro)):
        layer = new_macro[i]
        for j in range(len(layer)):
            ops_node = layer[j]
            ops = ops_node['ops']
            for k in range(len(ops)):
                if macro_pkl[i][j][k]:
                    new_macro[i][j]['ops'][k].build(input_shape=macro_pkl[i][j][k]['shape'])
                    new_macro[i][j]['ops'][k].set_weights(macro_pkl[i][j][k]['weights'])
                    # if isinstance(new_macro[i][j]['ops'][k])
                    # for weight in macro_pkl[i][j][k]:
                        #
                        # new_macro[i][j]['ops'][k].set_weights(macro_pkl[i][j][k])
    new_container['fc'].build(input_shape=container_fc_shape)
    new_container['fc'].set_weights(container_fc_weights)
    return new_macro, new_container

def train_macro(filename='testmacro.pkl'):

    batch_size = 128

    (xtrain, ytrain), (xtest, ytest) = prepare_data(dataset='mnist', batch_size=batch_size, shuffle=True)

    # out_channels = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256]
    out_channels = [8, 16, 32, 64, 128, 256]
    pool_size = (2, 2)
    # macro, container = restore_macro(filename='macrocontainer2.pkl', input_shape=xtrain[0].shape, out_channels=out_channels)

    macro, container = build_macro_graph(input_shape=xtrain[0].shape, out_channels=out_channels, regularizer='l2', regularization=1e-5)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05, decay_steps=10000, decay_rate=0.9, staircase=True)
    # optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=0.9, nesterov=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

    build_time = 0
    inference_time = 0
    gradients_time = 0
    apply_grads_time = 0
    rejections = 0

    epochs = 30
    batches = len(xtrain)
    # batches = 400
    for e in range(epochs):
        epoch_accuracy_fn.reset_states()
        np.random.RandomState(seed=e).shuffle(xtrain)
        np.random.RandomState(seed=e).shuffle(ytrain)

        for m in range(batches):
            start = time.time()
            print(f'epoch {e}, batch {m}')
            rejections -= 1
            while True:
                rejections += 1
                op_actions, connections = generate_random_architecture(len(out_channels))
                op_actions = [0, 0, 4, 1, 3, 3]
                connections = [[1], [0, 1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
                if validate_architecture(op_actions, connections, xtrain[m].shape, pool_size):
                    break
            sample_time = time.time() - start
            start = time.time()

            model = build_from_actions(macro, container, op_actions, connections)
            build_time += time.time() - start
            start = time.time()
            with tf.GradientTape() as tape:
                model_out = model(xtrain[m])
                loss = loss_fn(ytrain[m], model_out)
                accuracy_fn.update_state(ytrain[m], model_out)
                epoch_accuracy_fn.update_state(ytrain[m], model_out)
                accuracy = accuracy_fn.result()
                epoch_accuracy = epoch_accuracy_fn.result()
                accuracy_fn.reset_states()
                inference_time += time.time() - start
                start = time.time()
                gradients = tape.gradient(loss, model.trainable_variables)
            gradients_time += time.time() - start
            start = time.time()
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            apply_grads_time += time.time() - start
            print(f'\nsample={sample_time}\nbuild={build_time}\ninference={inference_time}\ngradients={gradients_time}\n'
                  f'apply={apply_grads_time}\nrejections={rejections}\nloss={loss}\naccuracy={accuracy}\n'
                  f'epoch accuracy={epoch_accuracy}\nop_actions={op_actions}\nconnections={connections}\n'
                  f'optimizer_step={optimizer.iterations.numpy()}\n'
                  f'learning_rate={lr_schedule.__call__(optimizer.iterations.numpy())}')

            # save_macro(macro, container, filename='macrocontainer.pkl')
        # save_macro(macro, container, filename='macrocontainer.pkl', template='macrocontainer.pkl')
        # macro, new_container = restore_macro(filename='macrocontainer.pkl', input_shape=xtrain[0].shape, out_channels=out_channels)

    print(model.summary())
    tf.keras.utils.plot_model(model, 'testmodel.png')





if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    train_macro()

    # graph_layers = build_macro_graph(input_shape=(128, 28, 28, 1))
    # train_model()
    print('test')


