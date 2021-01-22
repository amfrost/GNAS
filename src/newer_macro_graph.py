import math
import time
from functools import partial
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
print(tf.compat.v1.executing_eagerly())
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, AveragePooling2D, Input, Flatten, Dense, \
    Softmax, Activation, BatchNormalization, Layer
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
# import tensorflow.compat.v1 as v1
from tensorflow import compat as c
import numpy as np
import gnas_data as gdata
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


NHWC = 'NHWC'
CONV = 'conv'
SEP = 'sep'
AVG = 'avg'
MAX = 'max'
# OP_TYPES = ['conv3x3', 'maxpool']
# OP_TYPES = ['conv3x3']
OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
# OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5']

class MacroGraph(object):

    def __init__(self, input_shape, output_depths, depthwise_round='up', regularizer='l2', regularization_rate=1e-4,
                 pool_size=(2, 2), height_dim=1, width_dim=2, dropout_keep_prob=0.9):
        super(MacroGraph, self).__init__()
        self.input_shape = input_shape
        self.output_depths = output_depths
        self.num_layers = len(self.output_depths)
        input_depths = [input_shape[-1]]
        for i in range(len(output_depths) - 1):
            input_depths.append(output_depths[i])
        self.input_depths = input_depths
        self.depthwise_round = depthwise_round
        self.regularizer = regularizer
        self.regularization_rate = regularization_rate
        # self.graph_layers, self.containing_components = self.build_macro_graph()
        self.pool_size = pool_size
        self.height_dim = height_dim
        self.width_dim = width_dim
        # self.initializer = tf.initializers.GlorotUniform()
        self.initializer = tf.initializers.HeNormal()
        self.optimizer = partial(c.v1.train.AdamOptimizer, learning_rate=0.001)
        # self.optimizer = partial(c.v1.train.MomentumOptimizer, learning_rate=0.05, momentum=0.9,
        #                          use_locking=True, use_nesterov=True)
        self.dropout_keep_prob = dropout_keep_prob

        graph, initialise, train_step, logits, acc_metrics = self.build_macro_graph()
        self.graph = graph
        self.initialise = initialise
        self.train_step = train_step
        self.logits = logits
        self.macro_batch_acc = acc_metrics[0]
        self.ctrl_batch_acc = acc_metrics[1]
        self.macro_epoch_acc = acc_metrics[2]
        self.ctrl_epoch_acc = acc_metrics[3]

        self.sess = c.v1.Session(graph=self.graph)


    def update_accuracy(self, acc_tuple, logits, labels):
        self.sess.run(acc_tuple[1], {acc_tuple[2]: logits, acc_tuple[3]: labels})

    def retrieve_accuracy(self, acc_tuple):
        return self.sess.run(acc_tuple[0])

    def reset_accuracy(self, acc_tuple):
        self.sess.run(acc_tuple[4])


    def make_accurcy_metric(self, graph, scope_name):
        with c.v1.variable_scope(scope_name, reuse=c.v1.AUTO_REUSE):
            logits = c.v1.placeholder(tf.float32, shape=[None, 10], name='logits')
            labels = c.v1.placeholder(tf.uint8, shape=[None, ], name='labels')
            acc_fn, acc_update_fn = c.v1.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1),
                                                          name='metric')
            acc_vars = graph.get_collection(c.v1.GraphKeys.LOCAL_VARIABLES, scope=scope_name)
            acc_initialiser = c.v1.variables_initializer(acc_vars)
            acc_tuple = (acc_fn, acc_update_fn, logits, labels, acc_initialiser)
        return acc_tuple



    def build_macro_graph(self):
        g = c.v1.Graph()
        with g.as_default():
            optimizer = self.optimizer()

            macro_batch_acc_tuple = self.make_accurcy_metric(g, 'macro_batch_acc')
            ctrl_batch_acc_tuple = self.make_accurcy_metric(g, 'ctrl_batch_acc')
            macro_epoch_acc_tuple = self.make_accurcy_metric(g, 'macro_epoch_acc')
            ctrl_epoch_acc_tuple = self.make_accurcy_metric(g, 'ctrl_epoch_acc')
            acc_metrics = (macro_batch_acc_tuple, ctrl_batch_acc_tuple, macro_epoch_acc_tuple, ctrl_epoch_acc_tuple)

            with c.v1.variable_scope('input', reuse=c.v1.AUTO_REUSE):
                x = c.v1.placeholder(dtype=tf.float32, shape=self.input_shape, name='x')
                y = c.v1.placeholder(dtype=tf.int32, shape=(self.input_shape[0],), name='y')
                op_types = c.v1.placeholder(dtype=tf.int32, shape=(self.num_layers,), name='op_types')
                working_shapes = c.v1.placeholder(dtype=tf.int32, shape=(self.num_layers, 2), name='working_shapes')
                connections = []
                for i in range(self.num_layers):
                    connection_input = c.v1.placeholder(dtype=tf.uint8, shape=(i,), name=f'connections_layer_{i}')
                    connections.append(connection_input)

            with c.v1.variable_scope(f'layer_{0}', reuse=c.v1.AUTO_REUSE):
                self.generate_layer_vars(0, x)
                out = self.apply_layer(x, 0, op_types, working_shapes)
                out = tf.reshape(out, shape=(
                    out.shape[0], working_shapes[0][0], working_shapes[0][1], self.output_depths[0]))
                out = self.apply_batchnorm(out, '0')
                out = tf.nn.relu(out)

            layer_inputs = [out]

            for layer_index in range(1, self.num_layers):
                with c.v1.variable_scope(f'layer_{layer_index}', reuse=c.v1.AUTO_REUSE):
                    self.generate_factorised_reduction_vars(layer_index)
                    out = self.apply_consolidation(layer_index, connections, working_shapes, layer_inputs)

                    out = tf.reshape(out, shape=(out.shape[0], working_shapes[layer_index-1][0],
                                                 working_shapes[layer_index-1][1], self.input_depths[layer_index]))
                    out = self.apply_batchnorm(out, '0')
                    out = tf.nn.relu(out)

                    self.generate_layer_vars(layer_index, out)
                    out = self.apply_layer(out, layer_index, op_types, working_shapes)
                    out = tf.reshape(out, shape=(out.shape[0], working_shapes[layer_index][0],
                                                 working_shapes[layer_index][1], self.output_depths[layer_index]))

                    out = self.apply_batchnorm(out, '1')
                    out = tf.nn.relu(out)

                    layer_inputs.append(out)


            with c.v1.variable_scope('fc0', reuse=c.v1.AUTO_REUSE):
                weights = c.v1.get_variable('weights',
                                            initializer=self.initializer(shape=(self.output_depths[-1], 10),
                                                                         dtype=tf.float32),
                                            regularizer=lambda w: tf.reduce_mean(w ** 2),
                                            trainable=True)
                bias = c.v1.get_variable('bias',
                                         initializer=self.initializer([10]),
                                         trainable=True)

            out = layer_inputs[-1]
            out = tf.reduce_mean(out, axis=[1, 2])
            out = tf.nn.dropout(out, self.dropout_keep_prob)
            out = tf.reshape(out, shape=(tf.shape(out)[0], -1))
            out = tf.matmul(out, weights)
            out = tf.add(out, bias)

            logits = tf.identity(out, name='logits')
            prediction = tf.nn.softmax(out, name='prediction')
            cross_entropy = tf.losses.sparse_categorical_crossentropy(y, prediction, from_logits=False)
            loss = tf.reduce_mean(cross_entropy, name='loss')

            # reg_losses = c.v1.losses.get_regularization_losses()
            # reg_loss = c.v1.losses.get_regularization_loss()

            squared_grads = [tf.square(g) for g in g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)]
            mean_squares = [tf.reduce_mean(sg) for sg in squared_grads]
            mean_reg_losses = tf.reduce_mean(mean_squares)

            loss += self.regularization_rate * mean_reg_losses


            # loss += self.regularization_rate * reg_loss
            gradients = c.v1.gradients([loss], g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))
            gradients = [tf.clip_by_norm(g, 5.0) for g in gradients]
            train_step = optimizer.apply_gradients(zip(gradients, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)))
            init = c.v1.global_variables_initializer()

        return g, init, train_step, logits, acc_metrics

    def generate_layer_vars(self, layer_index, input_tensor):
        shape = (3, 3, self.input_depths[layer_index], self.output_depths[layer_index])
        self.get_variable('filters', f'{CONV}{shape[0]}x{shape[1]}', shape)
        shape = (5, 5, self.input_depths[layer_index], self.output_depths[layer_index])
        self.get_variable('filters', f'{CONV}{shape[0]}x{shape[1]}', shape)

        shape = (3, 3, self.input_depths[layer_index], self.get_depthwise_multiplier(layer_index))
        output_depth = self.output_depths[layer_index]
        pw_shape = (1, 1, shape[2] * shape[3], output_depth)
        self.get_variable('dw_filters', f'{SEP}{shape[0]}x{shape[1]}', shape)
        self.get_variable('pw_filters', f'{SEP}{shape[0]}x{shape[1]}', pw_shape)

        shape = (5, 5, self.input_depths[layer_index], self.get_depthwise_multiplier(layer_index))
        output_depth = self.output_depths[layer_index]
        pw_shape = (1, 1, shape[2] * shape[3], output_depth)
        self.get_variable('dw_filters', f'{SEP}{shape[0]}x{shape[1]}', shape)
        self.get_variable('pw_filters', f'{SEP}{shape[0]}x{shape[1]}', pw_shape)

        self.get_variable('pw_filters', f'maxpool_{layer_index}',
                          shape=(1, 1, input_tensor.shape[-1], self.output_depths[layer_index]))

        self.get_variable('pw_filters', f'avgpool_{layer_index}',
                          shape=(1, 1, input_tensor.shape[-1], self.output_depths[layer_index]))

    def get_variable(self, var_name, scope_name, shape, dtype=tf.float32, initialiser=None,
                     reuse=c.v1.AUTO_REUSE, regulariser=None, trainable=True):
        # if initialiser is None:
        initialiser = self.initializer if initialiser is None else initialiser
        # if regulariser is None:
        regulariser = lambda w: tf.reduce_mean(w**2) if regulariser is None else regulariser

        with c.v1.variable_scope(scope_name, reuse=reuse):
            return c.v1.get_variable(var_name, initializer=initialiser(shape, dtype),
                                     regularizer=regulariser, trainable=trainable)

    @tf.function
    def apply_layer(self, input_tensor, layer_index, op_types, working_shapes):
        op = op_types[layer_index]
        if tf.equal(op, tf.cast(tf.constant(0), tf.int32)):
            out = self.apply_conv2d(shape=(3, 3, self.input_depths[layer_index], self.output_depths[layer_index]),
                                    input_tensor=input_tensor)
        elif tf.equal(op, tf.cast(tf.constant(1), tf.int32)):
            out = self.apply_conv2d(shape=(5, 5, self.input_depths[layer_index], self.output_depths[layer_index]),
                                    input_tensor=input_tensor)

        elif tf.equal(op, tf.cast(tf.constant(2), tf.int32)):
            out = self.apply_sep_conv2d(
                shape=(3, 3, self.input_depths[layer_index], self.get_depthwise_multiplier(layer_index)),
                output_depth=self.output_depths[layer_index], input_tensor=input_tensor)
        elif tf.equal(op, tf.cast(tf.constant(3), tf.int32)):
            out = self.apply_sep_conv2d(
                shape=(5, 5, self.input_depths[layer_index], self.get_depthwise_multiplier(layer_index)),
                output_depth=self.output_depths[layer_index], input_tensor=input_tensor)

        elif tf.equal(op, tf.cast(tf.constant(4), tf.int32)):
            out = self.apply_pool(layer_index=layer_index, type=AVG, ksize=2, input_tensor=input_tensor,
                                  working_shapes=working_shapes)
        elif tf.equal(op, tf.cast(tf.constant(5), tf.int32)):
            out = self.apply_pool(layer_index=layer_index, type=MAX, ksize=2, input_tensor=input_tensor,
                                  working_shapes=working_shapes)
        else:
            out = input_tensor
            # raise NotImplementedError()
        return out

    def apply_conv2d(self, shape, input_tensor):
        filters = self.get_variable('filters', f'{CONV}{shape[0]}x{shape[1]}', shape)
        return tf.nn.conv2d(input_tensor, filters, strides=1, padding="SAME", data_format=NHWC)

    def apply_sep_conv2d(self, shape, output_depth, input_tensor):
        depthwise_filters = self.get_variable('dw_filters', f'{SEP}{shape[0]}x{shape[1]}', shape)
        pw_shape = (1, 1, shape[2] * shape[3], output_depth)
        pointwise_filters = self.get_variable('pw_filters', f'{SEP}{shape[0]}x{shape[1]}', pw_shape)
        return tf.nn.separable_conv2d(input_tensor, depthwise_filters, pointwise_filters,
                                      strides=[1, 1, 1, 1], padding="SAME", data_format=NHWC)

    def apply_pool(self, layer_index, type, ksize, input_tensor, working_shapes):

        pointwise_filters = self.get_variable('pw_filters', f'{type}pool_{layer_index}', shape=(
            1, 1, self.input_depths[layer_index], self.output_depths[layer_index]))
        out = tf.nn.conv2d(input_tensor, pointwise_filters, strides=1, padding="SAME", data_format=NHWC)


        if type == AVG:
            out = tf.nn.avg_pool2d(out, ksize, strides=ksize, padding="VALID",
                                   data_format=NHWC, name=f'{type}pool')
        elif type == MAX:
            out = tf.nn.max_pool2d(out, ksize, strides=ksize, padding="VALID",
                                   data_format=NHWC, name=f'{type}pool')
        else:
            raise NotImplementedError()

        # paddings = [[0, 0],
        #             [0, (input_tensor.shape[1] - (input_tensor.shape[1] // ksize))],
        #             [0, (input_tensor.shape[2] - (input_tensor.shape[2] // ksize))],
        #             [0, 0]]

        # new_size = working_shapes[layer_index - 1] // ksize
        # pad_size = (working_shapes[layer_index - 1] - (working_shapes[layer_index - 1] // ksize))

        # tis_layer_working_size = working_shapes[layer_index]

        # paddings = [[0, 0],
        #             [0, (working_shapes[layer_index][0] - (working_shapes[layer_index - 1][0] // ksize))],
        #             [0, (working_shapes[layer_index][1] - (working_shapes[layer_index - 1][1] // ksize))],
        #             [0, 0]]
        # out = tf.pad(out, paddings)

        return out

    def apply_batchnorm(self, input_tensor, id):
        with c.v1.variable_scope(f'batchnorm_{id}', reuse=c.v1.AUTO_REUSE):
            mu = c.v1.get_variable('mu', initializer=tf.constant_initializer(0.0)([input_tensor.shape[-1]]),
                                   trainable=False)
            sigma = c.v1.get_variable('sigma', initializer=tf.constant_initializer(1.0)([input_tensor.shape[-1]]),
                                      trainable=False)
            offset = c.v1.get_variable('offset', initializer=tf.constant_initializer(0.0)([input_tensor.shape[-1]]),
                                       trainable=True)
            scale = c.v1.get_variable('scale', initializer=tf.constant_initializer(1.0)([input_tensor.shape[-1]]),
                                      trainable=True)
        out = tf.nn.batch_normalization(input_tensor, mu, sigma, offset, scale, variance_epsilon=0.001)
        return out

    def apply_consolidation(self, layer_index, connections, working_shapes, layer_inputs):
        active_inputs = connections[layer_index]
        consolidation_filters = self.get_variable('filters', f'consolidator_{layer_index}',
                                                  (1, 1, sum(self.output_depths[:layer_index]),
                                                  # (1, 1, len(layer_inputs) * self.input_depths[layer_index],
                                                   self.input_depths[layer_index]))

        return self.consolidate_inputs(layer_index, layer_inputs, active_inputs, working_shapes, consolidation_filters)

    @tf.function  # TODO: with the range being over the length of the layer inputs, does this need to be tf.function? if moving the if-condition elsewhere
    def consolidate_inputs(self, layer_index, layer_inputs, connection, working_shapes, filters):
        if tf.equal(tf.cast(tf.reduce_sum(connection), tf.int32), tf.constant(1, dtype=tf.int32)):
            return layer_inputs[-1]

        new_inputs = layer_inputs.copy()
        for target_index in range(len(layer_inputs)):

            new_input = layer_inputs[target_index] * tf.cast(connection[target_index], tf.float32)
            new_input = self.handle_required_pooling(new_input, working_shapes, target_index, layer_index)
            new_inputs[target_index] = new_input

        # new_inputs = tf.concat(new_inputs, axis=3)
        # new_inputs = tf.nn.conv2d(new_inputs, filters, strides=1, padding="SAME", data_format=NHWC)

        new_inputs = tf.add_n(new_inputs)

        return new_inputs

    def generate_factorised_reduction_vars(self, layer_index):
        with c.v1.variable_scope(f'fact_reduce_L{layer_index}'):
            for target_index in range(0, layer_index):
                self.get_variable('pw_filters_path1-1', f'from_{target_index}',
                                  shape=(1, 1, self.output_depths[target_index], self.output_depths[target_index] // 2))
                # self.get_variable('pw_filters_path1-2', f'from_{target_index}',
                #                   shape=(1, 1, self.input_depths[layer_index], self.input_depths[layer_index] // 2))
                self.get_variable('pw_filters_path2-1', f'from_{target_index}',
                                  shape=(1, 1, self.output_depths[target_index], self.output_depths[target_index] // 2))
                # self.get_variable('pw_filters_path2-2', f'from_{target_index}',
                #                   shape=(1, 1, self.input_depths[layer_index], self.input_depths[layer_index] // 2))
                self.get_variable('batchnorm_mu', f'from_{target_index}', initialiser=tf.constant_initializer(0.0),
                                  shape=([self.output_depths[target_index]]), trainable=False)
                self.get_variable('batchnorm_sigma', f'from_{target_index}', initialiser=tf.constant_initializer(1.0),
                                  shape=([self.output_depths[target_index]]), trainable=False)
                self.get_variable('batchnorm_offset', f'from_{target_index}', initialiser=tf.constant_initializer(0.0),
                                  shape=([self.output_depths[target_index]]), trainable=True)
                self.get_variable('batchnorm_scale', f'from_{target_index}', initialiser=tf.constant_initializer(1.0),
                                  shape=([self.output_depths[target_index]]), trainable=True)

    def factor_reduce_input(self, input_tensor, layer_index, target_index, ammend_channels=False):
        with c.v1.variable_scope(f'fact_reduce_L{layer_index}'):
            pw_filters_path1_a = self.get_variable('pw_filters_path1-1', f'from_{target_index}',
                                                 shape=(1, 1, self.output_depths[target_index],
                                                        self.output_depths[target_index] // 2))
                                                 # shape=(1, 1, self.output_depths[target_index],
                                                 #        self.input_depths[layer_index] // 2))
            # pw_filters_path1_b = self.get_variable('pw_filters_path1-2', f'from_{target_index}',
            #                                      shape=(1, 1, self.input_depths[layer_index],
            #                                             self.input_depths[layer_index] // 2))
            pw_filters_path2_a = self.get_variable('pw_filters_path2-1', f'from_{target_index}',
                                                 shape=(1, 1, self.output_depths[target_index],
                                                        self.output_depths[target_index] // 2))
                                                 # shape=(1, 1, self.output_depths[target_index],
                                                 #        self.input_depths[layer_index] // 2))
            # pw_filters_path2_b = self.get_variable('pw_filters_path2-2', f'from_{target_index}',
            #                                      shape=(1, 1, self.input_depths[layer_index],
            #                                             self.input_depths[layer_index] // 2))
            mu = self.get_variable('batchnorm_mu', f'from_{target_index}',
                                   initialiser=tf.constant_initializer(0.0),
                                   shape=([self.output_depths[target_index]]), trainable=False)
            sigma = self.get_variable('batchnorm_sigma', f'from_{target_index}',
                                      initialiser=tf.constant_initializer(1.0),
                                      shape=([self.output_depths[target_index]]), trainable=False)
            offset = self.get_variable('batchnorm_offset', f'from_{target_index}',
                                       initialiser=tf.constant_initializer(0.0),
                                       shape=([self.output_depths[target_index]]), trainable=True)
            scale = self.get_variable('batchnorm_scale', f'from_{target_index}',
                                      initialiser=tf.constant_initializer(1.0),
                                      shape=([self.output_depths[target_index]]), trainable=True)

        path1 = tf.nn.avg_pool2d(input_tensor, ksize=1, strides=2, padding="VALID", data_format=NHWC)
        # if ammend_channels:
        path1 = tf.nn.conv2d(path1, pw_filters_path1_a, strides=1, padding="SAME", data_format=NHWC)
        # else:
        #     path1 = tf.nn.conv2d(path1, pw_filters_path1_b, strides=1, padding="SAME", data_format=NHWC)

        pads = [[0, 0], [0, 1], [0, 1], [0, 0]]
        path2 = tf.pad(input_tensor, pads)[:, 1:, 1:, :]
        path2 = tf.nn.avg_pool2d(path2, ksize=1, strides=2, padding="VALID", data_format=NHWC)
        # if ammend_channels:
        path2 = tf.nn.conv2d(path2, pw_filters_path2_a, strides=1, padding="SAME", data_format=NHWC)
        # else:
        #     path2 = tf.nn.conv2d(path2, pw_filters_path2_b, strides=1, padding="SAME", data_format=NHWC)

        out = tf.concat([path1, path2], axis=3)
        out = tf.nn.batch_normalization(out, mu, sigma, offset, scale, variance_epsilon=0.001)
        return out



    @tf.function
    def handle_required_pooling(self, input_tensor, working_shapes, target_index, layer_index):
        # for j in range(num_pools):
        #     input_tensor = tf.nn.max_pool2d(input_tensor, ksize=ksize, strides=ksize, padding="VALID", data_format=NHWC)
        ksize = working_shapes[target_index][0] // working_shapes[layer_index - 1][0]
        # print(ksize)
        if tf.equal(ksize, tf.constant(1)):
            out = input_tensor
        elif tf.equal(ksize, tf.constant(2)):
            # out = tf.nn.max_pool2d(input_tensor, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            out = self.factor_reduce_input(input_tensor, layer_index, target_index, ammend_channels=True)
        elif tf.equal(ksize, tf.constant(4)):
            out = self.factor_reduce_input(input_tensor, layer_index, target_index, ammend_channels=True)
            out = self.factor_reduce_input(out, layer_index, target_index)
            # out = tf.nn.max_pool2d(input_tensor, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
        elif tf.equal(ksize, tf.constant(8)):
            out = self.factor_reduce_input(input_tensor, layer_index, target_index, ammend_channels=True)
            out = self.factor_reduce_input(out, layer_index, target_index)
            out = self.factor_reduce_input(out, layer_index, target_index)
            # out = tf.nn.max_pool2d(input_tensor, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
        elif tf.equal(ksize, tf.constant(16)):
            out = self.factor_reduce_input(input_tensor, layer_index, target_index, ammend_channels=True)
            out = self.factor_reduce_input(out, layer_index, target_index)
            out = self.factor_reduce_input(out, layer_index, target_index)
            out = self.factor_reduce_input(out, layer_index, target_index)
            # out = tf.nn.max_pool2d(input_tensor, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
        elif tf.equal(ksize, tf.constant(32)):
            out = self.factor_reduce_input(input_tensor, layer_index, target_index, ammend_channels=True)
            out = self.factor_reduce_input(out, layer_index, target_index)
            out = self.factor_reduce_input(out, layer_index, target_index)
            out = self.factor_reduce_input(out, layer_index, target_index)
            out = self.factor_reduce_input(out, layer_index, target_index)
            # out = tf.nn.max_pool2d(input_tensor, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
            # out = tf.nn.max_pool2d(out, ksize=2, strides=2, padding="VALID", data_format=NHWC)
        else:
            out = input_tensor

        return out

    def get_depthwise_multiplier(self, layer_index):
        """
        Tries to get multiplier closest to desired output channels. Will round up or down (see __init__).
        :param layer_index: Corresponds to depth of component in architecture
        :return: int
        """
        if self.depthwise_round == 'down':
            return math.floor(self.output_depths[layer_index] / self.input_depths[layer_index])
        elif self.depthwise_round == 'up':
            return math.ceil(self.output_depths[layer_index] / self.input_depths[layer_index])
        else:
            raise ValueError("Depthwise rounding must be either 'down' or 'up'.")

    def validate_op_types(self, op_types):
        num_pool_ops = len([op for op in op_types if op == 4 or op == 5])
        input_arr = np.array(self.input_shape)
        for i in range(num_pool_ops):
            input_arr[self.height_dim] //= self.pool_size[0]
            input_arr[self.width_dim] //= self.pool_size[1]
        return input_arr[self.height_dim] > 0 and input_arr[self.width_dim] > 0

    @staticmethod
    def create_feed_dict(batch, labels, op_types, connections):
        feed_dict = {'input/x:0': batch, 'input/y:0': labels, 'input/op_types:0': op_types}
        for i in range(len(connections)):
            con = connections[i]
            if len(con) > 0:
                con[-1] = 1
            # con[0] = 0
            feed_dict[f'input/connections_layer_{i}:0'] = con

        working_shapes = []
        cur_working_shape = [batch.shape[1], batch.shape[2]]
        for op in op_types:
            if op == 4 or op == 5:
                cur_working_shape[0] //= 2
                cur_working_shape[1] //= 2
            working_shapes.append((cur_working_shape[0], cur_working_shape[1]))

        feed_dict['input/working_shapes:0'] = working_shapes
        return feed_dict



if __name__ == '__main__':
    # output_depths = [8, 16, 32, 64, 128, 256]
    # output_depths = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256]
    # output_depths = [36] * 12
    # output_depths = [36] * 6
    # output_depths = [36] * 4
    output_depths = [8, 16, 32, 64]

    # op_types_in = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # op_types_in = [0] * 12
    # op_types_in = [0] * 3
    # op_types_in = [1, 4, 1, 1, 3, 1, 1, 1, 3, 2, 4, 1]
    # op_types_in = [1, 4, 4, 1, 3, 1]
    # op_types_in = [1, 3, 3, 2, 3, 0]
    op_types_in = [1, 4, 4, 1]
    # connections_in = [[1], [0, 1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
    # connections_in = [[1], [0, 1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
    # connections_in = [[1], [0, 1], [1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 1]]
    connections_in = [[], [1], [0, 1], [1, 0, 1]]
    # connections_in = [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    # connections_in = [
    #     [1],
    #     [1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # ]
    # connections_in = [
    #     [1],
    #     [0, 1],
    #     [0, 0, 1],
    #     [0, 0, 0, 1],
    #     [0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # ]
    # connections_in = [
    #     [1],
    #     [1, 1],
    #     [0, 1, 1],
    #     [0, 0, 1, 1],
    #     [0, 1, 1, 0, 1],
    #     [0, 0, 1, 0, 1, 1],
    #     [0, 1, 0, 0, 1, 1, 1],
    #     [0, 1, 0, 0, 0, 1, 1, 1],
    #     [0, 1, 0, 0, 0, 1, 0, 1, 1],
    #     [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    #     [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    # ]



    batch_size = 128
    xtrain_batches, ytrain_batches = gdata.prepare_data('cifar10', batch_size=batch_size, valid_size=None)
    macrograph = MacroGraph(input_shape=xtrain_batches[0].shape, output_depths=output_depths)

    g, init, train_step, logits, acc_metrics = macrograph.build_macro_graph()

    epochs = 100

    with c.v1.Session(graph=g) as sess:
        tf.compat.v1.enable_eager_execution()
        accuracy_logits = c.v1.placeholder(tf.float32, shape=[batch_size, 10])
        accuracy_labels = c.v1.placeholder(tf.uint8, shape=[batch_size, ])
        accuracy, accuracy_fn = c.v1.metrics.accuracy(labels=accuracy_labels,
                                                      predictions=tf.argmax(accuracy_logits, axis=1),
                                                      name='accuracy')
        accuracy_vars = c.v1.get_collection(c.v1.GraphKeys.LOCAL_VARIABLES, scope='accuracy')

        sess.run(init)
        sess.run(c.v1.local_variables_initializer())



        for e in range(epochs):
            start = time.time()
            np.random.RandomState(seed=e).shuffle(xtrain_batches)
            np.random.RandomState(seed=e).shuffle(ytrain_batches)
            for b in range(len(xtrain_batches)):
                batch_start = time.time()
                # while True:
                #     op_types_in = list(np.random.choice(6, 3))
                #     if macrograph.validate_op_types(op_types_in):
                #         break
                # print(op_types_in)
                feed_dict = macrograph.create_feed_dict(xtrain_batches[b], ytrain_batches[b], op_types_in, connections_in)
                fetches = [train_step, logits]
                # print(f'executing eager 0 = {c.v1.executing_eagerly()}')
                _, logits_out = sess.run(fetches, feed_dict)
                acc = sess.run([accuracy, accuracy_fn],
                               {accuracy_logits: logits_out, accuracy_labels: ytrain_batches[b]})
                if b % 50 == 0:
                    print(f'n-hundredth batch accuracy = {acc}, time={time.time() - batch_start}')
            print(f'############################accuracy = {acc}, time={time.time() - start}')
            sess.run(c.v1.variables_initializer(accuracy_vars))
