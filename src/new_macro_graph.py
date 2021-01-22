import math
import time
from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, AveragePooling2D, Input, Flatten, Dense, \
    Softmax, Activation, BatchNormalization, Layer
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
# import tensorflow.compat.v1 as v1
from tensorflow import compat as c
import numpy as np

NHWC = 'NHWC'
OP_TYPES = ['conv3x3', 'maxpool']
# OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
# OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5']

class MacroGraph(object):

    def __init__(self, input_shape, output_depths, depthwise_round='up', regularizer='l2', regularization_rate=1e-5,
                 pool_size=(2, 2), height_dim=1, width_dim=2):
        super(MacroGraph, self).__init__()
        self.input_shape = input_shape
        self.output_depths = output_depths
        input_depths = [input_shape[-1]]
        for i in range(len(output_depths) - 1):
            input_depths.append(output_depths[i])
        self.input_depths = input_depths
        self.depthwise_round = depthwise_round
        self.regularizer = regularizer
        self.regularization_rate = regularization_rate
        self.graph_layers, self.containing_components = self.build_macro_graph()
        self.pool_size = pool_size
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.initzr = tf.initializers.GlorotUniform()

    def get_depthwise_multiplier(self, layer_index):
        """
        Tries to get multiplier closest to desired output channels. Will round up or down (see __init__).
        :param layer_index: Corresponds to depth of component in architecture
        :return: int
        """
        if self.depthwise_round == 'up':
            return math.ceil(self.output_depths[layer_index] / self.input_depths[layer_index])
        elif self.depthwise_round == 'down':
            return math.floor(self.output_depths[layer_index] / self.input_depths[layer_index])
        else:
            raise ValueError("Depthwise rounding must be either 'down' or 'up'.")

    def make_regularizer(self):
        """
        Creates a keras regulariser. For 'l1l2', set rate in __init__ as (l1_rate, l2_rate)
        :return: keras regulariser
        """
        if self.regularizer is None:
            return None
        elif self.regularizer == 'l2':
            return tf.keras.regularizers.L2(l2=self.regularization_rate)
        elif self.regularizer == 'l1':
            return tf.keras.regularizers.L1(l1=self.regularization_rate)
        elif self.regularizer == 'l1l2':
            return tf.keras.regularizers.L1L2(l1=self.regularization_rate[0], l2=self.regularization_rate[1])
        else:
            raise NotImplementedError("Specify either 'l1' or 'l2' regularization.")

    def make_conv_ops(self, conv_layer, layer_index):
        return [
            GNASConcat(MaxPool2D(pool_size=(2, 2), padding='valid'), output_depth=self.input_depths[layer_index]),
            Conv2D(filters=self.input_depths[layer_index], kernel_size=(1, 1),
                   kernel_regularizer=self.make_regularizer(), bias_regularizer=self.make_regularizer()),
            Activation(relu),
            BatchNormalization(),
            conv_layer,
            Activation(relu)
        ]

    def make_pool_ops(self, pool_layer, layer_index):
        return [
            GNASConcat(MaxPool2D(pool_size=(2, 2), padding='valid'),
                       output_depth=self.output_depths[layer_index]),
            # tf.keras.layers.Conv2D(filters=out_channels[layer], kernel_size=(1, 1)),
            # tf.keras.layers.Layer(),  # so skip the same number of layers as conv ops
            Conv2D(filters=self.input_depths[layer_index], kernel_size=(1, 1),
                                   kernel_regularizer=self.make_regularizer(),
                                   bias_regularizer=self.make_regularizer()),
            Activation(relu),
            pool_layer,
        ]

    def get_weight(self, initializer, shape, name, trainable=True):
        w = tf.Variable(initializer(shape), name=name, trainable=trainable, dtype=tf.float32)
        self.weights.append(w)
        return w


    def init_weights(self):
        self.weights = []
        initializer = tf.initializers.glorot_uniform()

        weight_vals = [
            (initializer, (3, 3, 3, 16), 'conv2d1', True),
            (tf.constant_initializer(0.0), (16,), 'bnm1', True),
            (tf.constant_initializer(1.0), (16,), 'bnv1', True),
            (tf.constant_initializer(0.0), (16,), 'bno1', False),
            (tf.constant_initializer(1.0), (16,), 'bns1', False),
            (initializer, (3, 3, 16, 32), 'conv2d2', True),
            (tf.constant_initializer(0.0), (32,), 'bnm2', True),
            (tf.constant_initializer(1.0), (32,), 'bnv2', True),
            (tf.constant_initializer(0.0), (32,), 'bno2', False),
            (tf.constant_initializer(1.0), (32,), 'bns2', False),
            (initializer, (32 * 16 * 16, 10), 'fc1', True)
        ]

        self.weights = [self.get_weight(wv[0], wv[1], wv[2], wv[3]) for wv in weight_vals]


    def create_macro_model(self, sample_input, op_types, connections):

        x = tf.cast(sample_input, tf.float32)
        layer_inputs = [x]


        for layer_index in range(len(self.output_depths)):

            # with tf.compat.v1.VariableScope()
            #
            # consolidated_inputs = self.consolidate_inputs(layer_inputs, connections[layer_index])



            op_type = op_types[layer_index]
            op = self.get_op(op_type, layer_index)

    def consolidate_inputs(self, layer_inputs, connections):
        if sum(connections) == 1 and connections[-1] == 1:
            return layer_inputs[-1]

        output_depth = layer_inputs[-1].shape[-1]
        for i in range(len(connections)):
            if connections[i] == 0:
                layer_inputs[i] = tf.zeros_like(layer_inputs[i])
        output = tf.concat(layer_inputs, axis=-1)
        cur_depth = output.shape[-1]

        tf.nam

        pointwise_filters = tf.Variable(self.initzr(shape=(1, 1, cur_depth, output_depth)), trainable=True, dtype=tf.float32)
        output = tf.nn.conv2d(output, filters=pointwise_filters, strides=1, padding="SAME", data_format=NHWC)

        return output


    def get_op(self, op_type, layer_index):
        if op_type == 'conv3x3':

            op = tf.nn.conv2d()


    # @tf.function
    def build_tf_macro_graph(self, input_shape, op_types=None, connections=None):
        g = c.v1.Graph()
        with g.as_default():
            opt = c.v1.train.AdamOptimizer(learning_rate=0.001)

            with c.v1.variable_scope('gnas', reuse=c.v1.AUTO_REUSE):
                x = c.v1.placeholder(dtype=c.v1.float32, shape=input_shape, name='x')
                y = c.v1.placeholder(dtype=c.v1.int8, shape=(input_shape[0], ), name='y')


                with c.v1.variable_scope('layer_1', reuse=c.v1.AUTO_REUSE):
                    filters = c.v1.get_variable('conv3x3',
                                                initializer=self.initzr(shape=(3, 3, 3, 16), dtype=tf.float32),
                                                regularizer=lambda w: tf.reduce_mean(w**2),
                                                trainable=True)
                    out = tf.nn.conv2d(x, filters, strides=1, padding="SAME", data_format=NHWC)
                    out = tf.nn.relu(out)

                with c.v1.variable_scope('layer_2', reuse=c.v1.AUTO_REUSE):
                    with c.v1.variable_scope('batchnorm', reuse=c.v1.AUTO_REUSE):
                        mean = c.v1.get_variable('mean', initializer=tf.constant_initializer(0.0)([16]), trainable=False)
                        var = c.v1.get_variable('var', initializer=tf.constant_initializer(1.0)([16]), trainable=False)
                        offset = c.v1.get_variable('offset', initializer=tf.constant_initializer(0.0)([16]), trainable=True)
                        scale = c.v1.get_variable('scale', initializer=tf.constant_initializer(1.0)([16]), trainable=True)
                        out = tf.nn.batch_normalization(out, mean, var, offset, scale, variance_epsilon=0.001)
                    with c.v1.variable_scope('conv3x3', reuse=c.v1.AUTO_REUSE):
                        filters = c.v1.get_variable('filters', initializer=self.initzr(shape=(3, 3, 16, 32), dtype=tf.float32),
                                                    regularizer=lambda w: tf.reduce_mean(w**2),
                                                    trainable=True)
                        out = tf.nn.conv2d(out, filters, strides=1, padding="SAME", data_format=NHWC)
                        out = tf.nn.relu(out)
                    out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID")
                with c.v1.variable_scope('fc', reuse=False):
                    with c.v1.variable_scope('batchnorm', reuse=c.v1.AUTO_REUSE):
                        mean = c.v1.get_variable('mean', initializer=tf.constant_initializer(0.0)([32]), trainable=False)
                        var = c.v1.get_variable('var', initializer=tf.constant_initializer(1.0)([32]), trainable=False)
                        offset = c.v1.get_variable('offset', initializer=tf.constant_initializer(0.0)([32]), trainable=True)
                        scale = c.v1.get_variable('scale', initializer=tf.constant_initializer(1.0)([32]), trainable=True)
                        out = tf.nn.batch_normalization(out, mean, var, offset, scale, variance_epsilon=0.001)
                    weights = c.v1.get_variable('weights',
                                                initializer=self.initzr(shape=(32*16*16, 10), dtype=tf.float32),
                                                regularizer=lambda w: tf.reduce_mean(w**2),
                                                trainable=True)
                    bias = c.v1.get_variable('bias',
                                             initializer=self.initzr([10]),
                                             trainable=True)
                    out = tf.reshape(out, shape=(tf.shape(x)[0], -1))
                    out = tf.matmul(out, weights)
                    out = tf.add(out, bias)
                logits = tf.identity(out, name='logits')
                pred = tf.nn.softmax(out, name='pred')

                cross_entropy = tf.losses.sparse_categorical_crossentropy(y, pred, from_logits=False)
                loss = tf.reduce_mean(cross_entropy, name='loss')
                gradients = c.v1.gradients([loss], g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))
                gradients = opt.apply_gradients(zip(gradients, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)))
                init = c.v1.global_variables_initializer()



        return g, gradients, init, logits



        # x = tf.cast(sample_input, dtype=tf.float32)
        #
        #
        #
        # out = tf.nn.conv2d(x, filters=self.weights[0],
        #                    strides=[1, 1, 1, 1], padding="SAME", data_format=NHWC)
        # out = tf.nn.relu(out)
        # out = tf.nn.batch_normalization(out, mean=self.weights[1],
        #                                 variance=self.weights[2], offset=self.weights[3], scale=self.weights[4],
        #                                 variance_epsilon=0.001)
        # out = tf.nn.conv2d(out, filters=self.weights[5],
        #                    strides=[1, 1, 1, 1], padding="SAME", data_format=NHWC)
        # out = tf.nn.relu(out)
        # out = tf.nn.max_pool2d(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        #
        # out = tf.nn.batch_normalization(out, mean=self.weights[6],
        #                                 variance=self.weights[7], offset=self.weights[8], scale=self.weights[9],
        #                                 variance_epsilon=0.001)
        # out = tf.reshape(out, shape=(tf.shape(out)[0], -1))
        # out = tf.matmul(out, self.weights[10])
        # out = tf.nn.softmax(out)
        #
        # return out




    def build_better_macro_graph(self):

        sample_input_feed = tf.keras.layers.Input(shape=self.input_shape)

        # sample_input_feed = np.ones(shape=(128, 28, 28, 1), dtype=np.float32)

        current_output = sample_input_feed

        consolidator_input = tf.keras.layers.Input(shape=(1,))

        layer_consolidators = []

        required_inputs = [sample_input_feed]
        consolidator_inputs = []

        op_type_inputs = []
        connections_inputs = []

        layer_inputs = [current_output]

        for layer_index in range(len(self.output_depths)):

            layer_op_types_feed = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
            # layer_op_types_feed = tf.compat.v1.placeholder(dtype=tf.int8, shape=(1,))
            op_type_inputs.append(layer_op_types_feed)
            layer_connections_feed = tf.keras.layers.Input(shape=(layer_index + 1,), dtype=tf.int8)
            connections_inputs.append(layer_connections_feed)
            # layer_op_types_feed = np.array(1)
            # layer_connections_feed = np.array([1, 1])

            wrapper_ops = []
            for op_type in OP_TYPES:
                if op_type == 'conv3x3':
                    op = Conv2D(filters=self.output_depths[layer_index], kernel_size=(3, 3), padding='same')
                    # node_output = layer_wrapper_input([layer_wrapper_input, op, layer_inputs])
                    wrapper_ops.append(op)
                elif op_type == 'maxpool':
                    op = MaxPool2D(pool_size=(2, 2))
                    # wrapper = ShapeWrapper(op)
                    wrapper_ops.append(op)
                else:
                    raise NotImplementedError()

            fns = [partial(wrapper_ops[i], layer_inputs[0]) for i in range(len(OP_TYPES))]
            branch_fns = {k: v for (k, v) in enumerate(fns)}

            r = tf.case([(tf.equal(layer_op_types_feed, 0), fns[0]()), (tf.equal(layer_op_types_feed, 1), fns[1]())], default=fns[0])
            #
            r = tf.switch_case(tf.cast(tf.convert_to_tensor(layer_op_types_feed, dtype=tf.int32), tf.int32), branch_fns=branch_fns)
            #
            wrapper_output = r

            layer_wrapper = GnasLayer(layer_index, self.input_shape, self.output_depths, wrapper_ops)
            wrapper_output = layer_wrapper([layer_inputs, layer_op_types_feed, layer_connections_feed])

            layer_inputs.append(wrapper_output)

        fc = tf.keras.layers.Dense(10)
        model_out = fc(layer_inputs[-1])
        softmax = tf.keras.layers.Softmax()
        model_out = softmax(model_out)

        required_inputs.extend(op_type_inputs)
        required_inputs.extend(connections_inputs)

        model = tf.keras.models.Model(inputs=required_inputs, outputs=model_out)

        return model




    def build_macro_graph(self):
        graph_layers = []
        for layer_index in range(len(self.output_depths)):
            layer = []
            for op_type in OP_TYPES:
                layer_ops = {}
                layer_ops_id = f'{op_type}_L{layer_index}'
                layer_ops['id'] = layer_ops_id
                if op_type == 'conv3x3':
                    op = Conv2D(filters=self.output_depths[layer_index], kernel_size=(3, 3), padding='same',
                                kernel_regularizer=self.make_regularizer(), bias_regularizer=self.make_regularizer())
                    ops = self.make_conv_ops(op, layer_index)
                elif op_type == 'conv5x5':
                    op = Conv2D(filters=self.output_depths[layer_index], kernel_size=(5, 5), padding='same',
                                kernel_regularizer=self.make_regularizer(), bias_regularizer=self.make_regularizer())
                    ops = self.make_conv_ops(op, layer_index)
                elif op_type == 'sep3x3':
                    op = SeparableConv2D(filters=self.output_depths[layer_index], kernel_size=(3, 3),
                                         depth_multiplier=self.get_depthwise_multiplier(layer_index), padding='same',
                                         kernel_regularizer=self.make_regularizer(),
                                         bias_regularizer=self.make_regularizer())
                    ops = self.make_conv_ops(op, layer_index)
                elif op_type == 'sep5x5':
                    op = SeparableConv2D(filters=self.output_depths[layer_index], kernel_size=(5, 5),
                                         depth_multiplier=self.get_depthwise_multiplier(layer_index), padding='same',
                                         kernel_regularizer=self.make_regularizer(),
                                         bias_regularizer=self.make_regularizer())
                    ops = self.make_conv_ops(op, layer_index)
                elif op_type == 'maxpool':
                    op = GNASPool(MaxPool2D(pool_size=(2, 2)), output_depth=self.output_depths[layer_index])
                    ops = self.make_pool_ops(op, layer_index)
                elif op_type == 'avgpool':
                    op = GNASPool(AveragePooling2D(pool_size=(2, 2)), output_depth=self.output_depths[layer_index])
                    ops = self.make_pool_ops(op, layer_index)
                else:
                    raise NotImplementedError(f"Found op type '{op_type}' but no corresponding implementation.")

                layer_ops['ops'] = ops
                layer.append(layer_ops)

            graph_layers.append(layer)

        containing_components = {
            'input_layer': Input(shape=self.input_shape[1:]),
            'final_concat': GNASConcat(MaxPool2D(pool_size=(2, 2)), output_depth=self.output_depths[-1]),
            'global_avg_pool': GlobalAveragePool(output_depth=self.output_depths[-1]),
            # 'final_batchnorm': tf.keras.layers.BatchNormalization(),
            'flatten': Flatten(),
            'fc': Dense(units=10, input_shape=(self.output_depths[-1],), kernel_regularizer=self.make_regularizer(),
                        bias_regularizer=self.make_regularizer()),
            'softmax': Softmax()
        }

        return graph_layers, containing_components

    def build_subgraph_components(self, op_types, connections, input_layer):
        components = [self.graph_layers[i][op_types[i]] for i in range(len(op_types))]
        layer_outputs = [input_layer]

        # n_multitensors used to handle skipping operations if only one non-zero tensor in output
        n_multitensor_ops = 3  # must match number of ops in make_conv_ops TODO: check make_pool_ops
        n_multitensor_skips = 0

        for layer_ops, connections in zip(components, connections):
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
                if n_multitensor_skips % n_multitensor_ops != 0:  # continue until we've skipped all the unnecessary ops
                    n_multitensor_skips += 1
                    continue
                if nonzero_inputs == 1 and isinstance(op, GNASConcat):
                    n_multitensor_skips += 1
                    out = layer_inputs[nonzero_index]
                    continue
                out = op(out)
            layer_outputs.append(out)

        return [out]

    def build_model(self, op_types, connections, print_summary=False, plot_model_filename=None):
        input_layer = self.containing_components['input_layer']

        model_out = self.build_subgraph_components(op_types, connections, input_layer)

        model_out = self.containing_components['global_avg_pool'](model_out[0])
        model_out = self.containing_components['fc'](model_out)
        model_out = self.containing_components['softmax'](model_out)

        model = Model(inputs=input_layer, outputs=model_out)

        if print_summary:
            print(model.summary())
        if plot_model_filename is not None:
            tf.keras.utils.plot_model(model, plot_model_filename)

        return model

    def validate_op_types(self, op_types):
        num_pool_ops = len([op for op in op_types if op == 4 or op == 5])
        input_arr = np.array(self.input_shape)
        for i in range(num_pool_ops):
            input_arr[self.height_dim] //= self.pool_size[0]
            input_arr[self.width_dim] //= self.pool_size[1]
        return input_arr[self.height_dim] > 0 and input_arr[self.width_dim] > 0


class GNASConcat(tf.keras.layers.Layer):
    def __init__(self, pool_layer: tf.keras.layers.Layer, output_depth, height_dim=1, width_dim=2, channel_dim=3):
        super(GNASConcat, self).__init__()
        self.pool_layer = pool_layer
        self.pool_size = pool_layer.pool_size
        self.output_depth = output_depth
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.channel_dim = channel_dim

    def poolable(self, start_dim, target_dim, pool_size):
        if self.pool_layer.padding != 'valid':
            raise NotImplementedError("Can only use GNASConcat with pooling padding = 'valid'.")
        while target_dim < start_dim:
            start_dim //= pool_size
        return start_dim == target_dim

    def find_target_dimension(self, input_tensors):
        shapes = np.array([np.asarray(it.shape) for it in input_tensors])
        smallest_height = np.min(shapes[:, self.height_dim])
        smallest_width = np.min(shapes[:, self.width_dim])
        for shape in shapes:
            assert self.poolable(shape[self.height_dim], smallest_height, self.pool_size[0]), \
                   f"Inputs with heights {shape[self.height_dim]}, {smallest_height} cannot be consolidated using " \
                   f"pool height = {self.pool_size[0]}"
            assert self.poolable(shape[self.width_dim], smallest_width, self.pool_size[1]), \
                   f"Inputs with widths {shape[self.width_dim]}, {smallest_width} cannot be consolidated using pool " \
                   f"width = {self.pool_size[1]}"
        return smallest_height, smallest_width

    def call(self, input_tensors, **kwargs):



        smallest_height, smallest_width = self.find_target_dimension(input_tensors)

        pooled_inputs = []
        for input_tensor in input_tensors:
            while input_tensor.shape[self.height_dim] > smallest_height:
                input_tensor = self.pool_layer(input_tensor)
            assert input_tensor.shape[self.height_dim] == smallest_height and \
                   input_tensor.shape[self.width_dim] == smallest_width, \
                   "Error while pooling input tensors for GNASConcat."
            pooled_inputs.append(input_tensor)

        return tf.concat(pooled_inputs, axis=self.channel_dim)


class GNASPool(tf.keras.layers.Layer):
    def __init__(self, pool_layer: tf.keras.layers.Layer, output_depth):
        super(GNASPool, self).__init__()
        self.pool_layer = pool_layer
        self.output_depth = output_depth
        self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.output_depth, kernel_size=(1, 1))

    def build(self, input_shape):
        input_shape_arr = np.array(input_shape)
        input_shape_arr[1] = input_shape[1] // self.pool_layer.pool_size[0]
        input_shape_arr[2] = input_shape[2] // self.pool_layer.pool_size[1]
        self.pointwise_conv.build(input_shape=input_shape_arr)

    def call(self, input_tensor, **kwargs):
        out = self.pool_layer(input_tensor)
        out = self.pointwise_conv(out)
        return out


class GlobalAveragePool(tf.keras.layers.Layer):
    def __init__(self, output_depth, height_dim=1, width_dim=2):
        super(GlobalAveragePool, self).__init__()
        self.output_depth = output_depth
        self.height_dim = height_dim
        self.width_dim = width_dim

    def call(self, input_tensors, **kwargs):
        return tf.reduce_mean(input_tensors, axis=[self.height_dim, self.width_dim])


class GnasLayer(tf.keras.layers.Layer):
    def __init__(self, layer_index, inp_shape, output_depth, wrapped_ops):
        super(GnasLayer, self).__init__()
        self.layer_index = layer_index
        self.inp_shape = inp_shape
        self.output_depth = output_depth
        self.wrapped_ops = wrapped_ops

        self.functions = []
        for wrapped_op in wrapped_ops:
            self.functions.append(partial(wrapped_op))


    def call(self, input_tensors, **kwargs):
        layer_inputs = input_tensors[0]
        layer_op_types_feed = input_tensors[1]
        layer_connections_feed = input_tensors[2]

        # fns = [partial(self.wrapped_ops[i], layer_inputs[0]) for i in range(len(OP_TYPES))]
        fns = [partial(self.wrapped_ops[i], layer_inputs[-1]) for i in range(len(OP_TYPES))]
        branch_fns = {k: v for (k, v) in enumerate(fns)}

        r = tf.switch_case(tf.cast(layer_op_types_feed, tf.int32), branch_fns=branch_fns, default=fns[0])

        print(r.shape)

        return r

class GnasWrapper(tf.keras.layers.Layer):
    def __init__(self, inp_shape, op):
        super(GnasWrapper, self).__init__()
        self.inp_shape = inp_shape
        self.op = op

    def call(self, input_tensor, **kwargs):
        return self.op(input_tensor)


import gnas_data as gdata
if __name__ == '__main__':
    output_depths = [8, 16, 32, 64]
    xtrain_batches, ytrain_batches = gdata.prepare_data('cifar10', 128, valid_size=None)
    macrograph = MacroGraph(input_shape=xtrain_batches[0][0].shape, output_depths=output_depths)
    g, train_step, init, logits = macrograph.build_tf_macro_graph(input_shape=xtrain_batches[0].shape)

    epochs = 5

    with c.v1.Session(graph=g) as sess:
        acc_logits = c.v1.placeholder(tf.float32, shape=[128, 10])
        acc_labels = c.v1.placeholder(tf.uint8, shape=[128, 1])
        acc, acc_op = c.v1.metrics.accuracy(labels=acc_labels, predictions=tf.argmax(acc_logits, axis=1), name='accscope')
        rs = c.v1.get_collection(c.v1.GraphKeys.LOCAL_VARIABLES, scope='accscope')

        sess.run(init)
        sess.run(c.v1.local_variables_initializer())
        for e in range(epochs):
            start = time.time()
            for b in range(len(xtrain_batches)):
                feed_dict = {'gnas/x:0': xtrain_batches[b], 'gnas/y:0': ytrain_batches[b][:, 0]}
                fetches = [train_step, logits]
                train_out, logits_out = sess.run(fetches=fetches, feed_dict=feed_dict)
                thisacc, thisop = sess.run([acc, acc_op], {acc_logits: logits_out, acc_labels: ytrain_batches[b]})
            print(thisacc)
            sess.run(c.v1.variables_initializer(rs))



class Consolidator(tf.keras.layers.Layer):
    def __init__(self, layer_id):
        super(Consolidator, self).__init__()
        self.layer_id = layer_id

    def call(self, input_tensor, **kwargs):
        pass

