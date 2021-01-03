import math
from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, AveragePooling2D, Input, Flatten, Dense, \
    Softmax, Activation, BatchNormalization, Layer
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
import numpy as np

OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
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
