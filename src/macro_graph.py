import math
import time
from functools import partial
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# print(tf.compat.v1.executing_eagerly())
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
# OP_TYPES = ['conv3x3', 'conv5x5']
OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
# OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5']
EPS = 1e-5

class MacroGraph(object):

    def __init__(self, input_shape, output_depths, depthwise_round='up', regularizer='l2', regularization_rate=0.0001,
                 pool_size=(2, 2), height_dim=1, width_dim=2, dropout_prob=0.1):
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
        self.initializer = tf.initializers.GlorotUniform()
        # self.initializer = tf.initializers.HeNormal()
        # self.optimizer = partial(c.v1.train.AdamOptimizer, learning_rate=0.00025)
        # self.optimizer = partial(c.v1.train.AdamOptimizer, learning_rate=0.0025)
        # self.optimizer = partial(c.v1.train.AdamOptimizer, learning_rate=0.001)
        # self.optimizer = partial(c.v1.train.MomentumOptimizer, learning_rate=0.04, momentum=0.9,
        #                          use_locking=False, use_nesterov=True)
        self.optimizer = c.v1.train.AdamOptimizer
        # self.dropout_keep_prob = dropout_keep_prob
        self.dropout_prob = 0.1

        graph, initialise, train_step, loss, logits, acc_metrics = self.build_macro_graph()
        self.graph = graph
        self.initialise = initialise
        self.train_step = train_step
        self.loss = loss
        self.logits = logits
        self.macro_batch_acc = acc_metrics[0]
        self.ctrl_batch_acc = acc_metrics[1]
        self.macro_epoch_acc = acc_metrics[2]
        self.ctrl_epoch_acc = acc_metrics[3]
        # self.optimizer = c.v1.train.AdamOptimizer(learning_rate=self.lr_out)

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


            macro_batch_acc_tuple = self.make_accurcy_metric(g, 'macro_batch_acc')
            ctrl_batch_acc_tuple = self.make_accurcy_metric(g, 'ctrl_batch_acc')
            macro_epoch_acc_tuple = self.make_accurcy_metric(g, 'macro_epoch_acc')
            ctrl_epoch_acc_tuple = self.make_accurcy_metric(g, 'ctrl_epoch_acc')
            acc_metrics = (macro_batch_acc_tuple, ctrl_batch_acc_tuple, macro_epoch_acc_tuple, ctrl_epoch_acc_tuple)


            with c.v1.variable_scope('g', reuse=c.v1.AUTO_REUSE):
                with c.v1.variable_scope('input'):
                    x = c.v1.placeholder(dtype=tf.float32, shape=self.input_shape, name='x')
                    y = c.v1.placeholder(dtype=tf.int32, shape=(self.input_shape[0],), name='y')
                    op_types = c.v1.placeholder(dtype=tf.int32, shape=(self.num_layers,), name='op_types')
                    working_shapes = c.v1.placeholder(dtype=tf.int32, shape=(self.num_layers, 2), name='working_shapes')
                    connections = []
                    for i in range(self.num_layers):
                        connection_input = c.v1.placeholder(dtype=tf.uint8, shape=(i,), name=f'connections_layer_{i}')
                        connections.append(connection_input)

                    learning_rate = c.v1.placeholder(dtype=tf.float32, shape=(), name='learning_rate')





                with c.v1.variable_scope(f'stem_conv'):
                    # layer_inputs = [x]
                    filters = self.get_variable('filters', 'stem', shape=(3, 3, x.shape[-1], self.output_depths[0]))
                    self.input_depths[0] = self.output_depths[0]
                    out = tf.nn.conv2d(x, filters, strides=1, padding="SAME", data_format=NHWC)
                    # out = self.apply_batchnorm(out, 'stem')

                with c.v1.variable_scope(f'layer_{0}'):
                    out = self.apply_batchnorm(out, '0')
                    out = tf.nn.relu(out)
                    out = self.apply_layer(out, 0, op_types)
                    out = tf.reshape(out, shape=(out.shape[0], working_shapes[0][1],
                                                 working_shapes[0][1], self.output_depths[0]))
                    layer_inputs = [out]

                for layer_index in range(1, self.num_layers):
                    with c.v1.variable_scope(f'layer_{layer_index}'):
                        out = layer_inputs[-1]
                        out = self.apply_batchnorm(out, 'pre')
                        out = tf.nn.relu(out)
                        out = self.apply_layer(out, layer_index, op_types)
                        layer_inputs.append(out)

                        # out = tf.cond(tf.equal(tf.cast(tf.reduce_sum(connections[layer_index]), tf.int32),
                        #                        tf.constant(0, tf.int32)),
                        #               lambda: out,
                        #               lambda: self.consolidate_residuals(
                        #                   layer_inputs, layer_index, connections, working_shapes))
                        out = self.consolidate_residuals(layer_inputs, layer_index, connections, working_shapes)

                        # out = self.consolidate_residuals(layer_inputs, layer_index, connections, working_shapes)
                        layer_inputs[-1] = out
                        if layer_index == 1 or layer_index == 3:
                            layer_inputs = self.factor_reduce(layer_inputs, layer_index)


                with c.v1.variable_scope('fc0'):
                    weights = c.v1.get_variable('weights', initializer=self.initializer(
                        shape=(self.output_depths[-1], 10), dtype=tf.float32), trainable=True)
                    bias = c.v1.get_variable('bias',
                                             initializer=tf.constant_initializer(0.0)([10]),
                                             trainable=True)

                    out = layer_inputs[-1]
                    out = tf.reduce_mean(out, axis=[1, 2])
                    out = tf.nn.dropout(out, self.dropout_prob)
                    out = tf.reshape(out, shape=(tf.shape(out)[0], -1))
                    out = tf.matmul(out, weights)
                    # out = tf.matmul(out, weights2)
                    out = tf.add(out, bias)

                logits = tf.identity(out, name='logits')
                prediction = tf.nn.softmax(out, name='prediction')
                cross_entropy = tf.losses.sparse_categorical_crossentropy(y, prediction, from_logits=False)
                # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                loss = tf.reduce_mean(cross_entropy, name='loss')

                # reg_losses = c.v1.losses.get_regularization_losses()
                # reg_loss = c.v1.losses.get_regularization_loss()

                squared_weights = [tf.square(g) for g in g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)]
                mean_squares = [tf.reduce_mean(sw) for sw in squared_weights]
                mean_reg_losses = tf.reduce_mean(mean_squares)

                loss += self.regularization_rate * mean_reg_losses


                gradients = c.v1.gradients([loss], g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))
                global_grad_norm = tf.linalg.global_norm(gradients)
                # gradients = [tf.clip_by_norm(g, global_grad_norm) for g in gradients]
                # gradients = [tf.clip_by_global_norm(g, global_grad_norm)[0] for g in gradients]
                optimizer = self.optimizer(learning_rate=learning_rate)
                train_step = optimizer.apply_gradients(zip(gradients, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)))
                # train_step = optimizer.minimize(loss)
                init = c.v1.global_variables_initializer()

        return g, init, train_step, loss, logits, acc_metrics

    def get_variable(self, var_name, scope_name, shape, dtype=tf.float32, initialiser=None,
                     reuse=c.v1.AUTO_REUSE, regulariser=None, trainable=True):
        initialiser = self.initializer if initialiser is None else initialiser
        with c.v1.variable_scope(scope_name):
            return c.v1.get_variable(var_name, initializer=lambda: initialiser(shape, dtype), trainable=trainable)

    def apply_layer(self, input_tensor, layer_index, op_types):
        op = op_types[layer_index]
        branch_fns = {}

        branch_fns[0] = lambda: self.apply_conv2d(
            shape=(3, 3, self.input_depths[layer_index], self.output_depths[layer_index]),
            input_tensor=input_tensor)
        branch_fns[1] = lambda: self.apply_conv2d(
            shape=(5, 5, self.input_depths[layer_index], self.output_depths[layer_index]),
            input_tensor=input_tensor)

        branch_fns[2] = lambda: self.apply_sep_conv2d(shape=(3, 3, self.input_depths[layer_index], 1),
                                                      output_depth=self.output_depths[layer_index],
                                                      input_tensor=input_tensor)
        branch_fns[3] = lambda: self.apply_sep_conv2d(shape=(5, 5, self.input_depths[layer_index], 1),
                                                      output_depth=self.output_depths[layer_index],
                                                      input_tensor=input_tensor)

        branch_fns[4] = lambda: self.apply_pool(layer_index=layer_index, type=AVG, ksize=3,
                                                input_tensor=input_tensor)
        branch_fns[5] = lambda: self.apply_pool(layer_index=layer_index, type=MAX, ksize=3,
                                                input_tensor=input_tensor)
        branch_fns[6] = lambda: input_tensor
        # with tf.device('/GPU:0'):
        r = tf.switch_case(tf.cast(op, tf.int32), branch_fns=branch_fns, default=branch_fns[6])
        return r

    def consolidate_residuals(self, layer_inputs, layer_index, connections, working_shapes):

        with c.v1.variable_scope(f'consolidate_L{layer_index}'):

            next_layer_inputs = layer_inputs.copy()

            connection = connections[layer_index]
            residuals = []
            for target_index in range(len(next_layer_inputs) - 1):
                new_input = tf.cond(tf.equal(tf.cast(connection[target_index], tf.int32), tf.constant(1, tf.int32)),
                                    lambda: next_layer_inputs[target_index],
                                    lambda: tf.zeros_like(next_layer_inputs[target_index]))
                # new_input = next_layer_inputs[target_index] * tf.cast(connection[target_index], tf.float32)
                residuals.append(new_input)
            residuals.append(next_layer_inputs[-1])

            # residual_filters = self.get_variable('filters', f'residuals_{layer_index}',
            #                                      (1, 1, sum(self.output_depths[:layer_index+1]),
            #                                       self.output_depths[layer_index]))
            # residuals = tf.concat(residuals, axis=3)
            # out = tf.nn.conv2d(residuals, residual_filters, strides=1, padding="SAME", data_format=NHWC)
            out = tf.add_n(residuals)

            out = tf.reshape(out, shape=(out.shape[0], working_shapes[layer_index-1][0],
                                         working_shapes[layer_index-1][1], self.output_depths[layer_index]))
            return out

    def factor_reduce(self, layer_inputs, layer_index):
        factor_reduced_layers = []
        for target_index in range(len(layer_inputs)):
            fr_layer = self.factor_reduce_input(layer_inputs[target_index], layer_index, target_index)
            factor_reduced_layers.append(fr_layer)
        return factor_reduced_layers

    def factor_reduce_input(self, input_tensor, layer_index, target_index):
        with c.v1.variable_scope(f'fact_reduce_L{layer_index}'):
            with c.v1.variable_scope(f'from_{target_index}'):
                shape = (1, 1, self.output_depths[target_index], self.output_depths[layer_index] // 2)
                pw_filters_path1 = c.v1.get_variable('pw_filters_path1',
                                                     initializer=lambda: self.initializer(shape, tf.float32),
                                                     regularizer=None, trainable=True)
                shape = (1, 1, self.output_depths[target_index], self.output_depths[layer_index] // 2)
                pw_filters_path2 = c.v1.get_variable('pw_filters_path2',
                                                     initializer=lambda: self.initializer(shape, tf.float32),
                                                     regularizer=None, trainable=True)

                path1 = tf.nn.avg_pool2d(input_tensor, ksize=1, strides=[1, 2, 2, 1], padding="VALID", data_format=NHWC)
                path1 = tf.nn.conv2d(path1, pw_filters_path1, strides=1, padding="SAME", data_format=NHWC)

                pads = [[0, 0], [0, 1], [0, 1], [0, 0]]
                path2 = tf.pad(input_tensor, pads)[:, 1:, 1:, :]
                path2 = tf.nn.avg_pool2d(path2, ksize=1, strides=[1, 2, 2, 1], padding="VALID", data_format=NHWC)
                path2 = tf.nn.conv2d(path2, pw_filters_path2, strides=1, padding="SAME", data_format=NHWC)

                out = tf.concat([path1, path2], axis=3)
                out = self.apply_batchnorm(out, f'fr_L{layer_index}_from_{target_index}')
                return out

    def apply_conv2d(self, shape, input_tensor):
        with c.v1.variable_scope(f'{CONV}{shape[0]}x{shape[1]}'):
            pointwise_filters = self.get_variable(f'filters', f'pointwise',
                                                  shape=(1, 1, shape[2], shape[3]))
            out = tf.nn.conv2d(input_tensor, pointwise_filters, strides=1, padding="SAME", data_format=NHWC)
            out = self.apply_batchnorm(out, '1')
            out = tf.nn.relu(out)

            conv_filters = self.get_variable(f'filters', f'conv_op',
                                             shape=(shape[0], shape[1], shape[3], shape[3]))
            out = tf.nn.conv2d(out, conv_filters, strides=1, padding="SAME", data_format=NHWC)
            return out

    def apply_sep_conv2d(self, shape, output_depth, input_tensor):
        with c.v1.variable_scope(f'{SEP}{shape[0]}x{shape[1]}'):
            pointwise_filters_0 = self.get_variable('pw_filters_0', f'{SEP}{shape[0]}x{shape[1]}',
                                                    shape=(1, 1, shape[2], output_depth))
            out = tf.nn.conv2d(input_tensor, pointwise_filters_0, strides=1, padding="SAME", data_format=NHWC)
            out = self.apply_batchnorm(out, '1')
            out = tf.nn.relu(out)

            depthwise_filters = self.get_variable('dw_filters', f'{SEP}{shape[0]}x{shape[1]}',
                                                  shape=(shape[0], shape[1], output_depth, 1))
            pointwise_filters_1 = self.get_variable('pw_filters_1', f'{SEP}{shape[0]}x{shape[1]}',
                                                  shape=(1, 1, output_depth, output_depth))
            out = tf.nn.separable_conv2d(out, depthwise_filters, pointwise_filters_1,
                                         strides=[1, 1, 1, 1], padding="SAME", data_format=NHWC)
            return out

    def apply_pool(self, layer_index, type, ksize, input_tensor):
        with c.v1.variable_scope(f'{type}pool'):
            pointwise_filters = self.get_variable('pw_filters', f'{type}pool_{layer_index}', shape=(
                1, 1, self.input_depths[layer_index], self.output_depths[layer_index]))
            out = tf.nn.conv2d(input_tensor, pointwise_filters, strides=1, padding="SAME", data_format=NHWC)
            out = self.apply_batchnorm(out, 'pool')

            if type == AVG:
                out = tf.nn.avg_pool2d(out, ksize, strides=1, padding="SAME",
                                       data_format=NHWC, name=f'{type}pool')
            elif type == MAX:
                out = tf.nn.max_pool2d(out, ksize, strides=1, padding="SAME",
                                       data_format=NHWC, name=f'{type}pool')
            else:
                raise NotImplementedError()
            return out

    def apply_batchnorm(self, input_tensor, id):
        with c.v1.variable_scope(f'batchnorm_{id}'):
            mu = c.v1.get_variable('mu', initializer=lambda: tf.constant_initializer(0.0)([input_tensor.shape[-1]]),
                                   trainable=False)
            sigma = c.v1.get_variable('sigma', initializer=lambda: tf.constant_initializer(1.0)([input_tensor.shape[-1]]),
                                      trainable=False)
            offset = c.v1.get_variable('offset', initializer=lambda: tf.constant_initializer(0.0)([input_tensor.shape[-1]]),
                                       trainable=True)
            scale = c.v1.get_variable('scale', initializer=lambda: tf.constant_initializer(1.0)([input_tensor.shape[-1]]),
                                      trainable=True)
        out = tf.nn.batch_normalization(input_tensor, mu, sigma, offset, scale, variance_epsilon=EPS)
        return out

    def validate_op_types(self, op_types):
        return True
        # num_pool_ops = len([op for op in op_types if op == 4 or op == 5])
        # input_arr = np.array(self.input_shape)
        # for i in range(num_pool_ops):
        #     input_arr[self.height_dim] //= self.pool_size[0]
        #     input_arr[self.width_dim] //= self.pool_size[1]
        # return input_arr[self.height_dim] > 0 and input_arr[self.width_dim] > 0

    @staticmethod
    def create_feed_dict(batch, labels, op_types, connections, learning_rate):
        feed_dict = {'g/input/x:0': batch, 'g/input/y:0': labels, 'g/input/op_types:0': op_types}
        for i in range(len(connections)):
            con = connections[i]
            # if len(con) > 0:
            #     con[-1] = 1
            # con[0] = 0
            feed_dict[f'g/input/connections_layer_{i}:0'] = con

        working_shapes = []
        cur_working_shape = [batch.shape[1], batch.shape[2]]
        for op in op_types:
            if op == 4 or op == 5:
                cur_working_shape[0] //= 2
                cur_working_shape[1] //= 2
            working_shapes.append((cur_working_shape[0], cur_working_shape[1]))

        working_shapes = []
        cur_working_shape = [batch.shape[1], batch.shape[2]]
        for i in range(len(op_types)):
            if i == 1 or i == 3:
                cur_working_shape[0] //= 2
                cur_working_shape[1] //= 2
            working_shapes.append((cur_working_shape[0], cur_working_shape[1]))

        feed_dict['g/input/working_shapes:0'] = working_shapes
        feed_dict['g/input/learning_rate:0'] = learning_rate
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
