from tensorflow import compat as c
import tensorflow as tf
from tensorflow import compat as c
import numpy as np
import gnas_data as gdata
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import math
from math import cos, pi, pow, e
import time
from time import process_time, perf_counter
import macro_graph as mg
import controller_gnn as ctrl
import pickle
import matplotlib.pyplot as plt
import pandas as pd

NHWC = 'NHWC'
CONV = 'conv'
SEP = 'sep'
AVG = 'avg'
MAX = 'max'
OP_TYPES = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'maxpool', 'avgpool']
EPS = 1e-5
class ArchitectureEval(object):
    def __init__(self, **kwargs):
        super(ArchitectureEval, self).__init__()
        # self.op_seq = op_seq
        # self.arc_seq = arc_seq
        # self.input_shape = kwargs.get('input_shape')
        self.output_dir = kwargs.get('output_dir')
        self.log_save_path = kwargs.get('log_save_path')
        self.initializer = kwargs.get('initializer')
        self.batch_size = kwargs.get('batch_size')
        self.dropout_prob = kwargs.get('dropout')
        self.regularization_rate = kwargs.get('regularization_rate')
        self.optimizer = kwargs.get('optimizer')
        # self.macro_graph_save_path = kwargs.get('macro_graph_save_path')
        # self.controller_graph_save_path = kwargs.get('controller_graph_save_path')
        self.checkpoint_id = kwargs.get('checkpoint_id')
        self.dataset = kwargs.get('dataset')
        self.valid_size = kwargs.get('valid_size')
        self.run_dir = kwargs.get('run_dir')
        self.macro_cosine_lr_init_period = kwargs.get('macro_cosine_lr_init_period')
        self.macro_cosine_lr_max = kwargs.get('macro_cosine_lr_max')
        self.macro_cosine_lr_min = kwargs.get('macro_cosine_lr_min')
        self.macro_cosine_lr_period_multiplier = kwargs.get('macro_cosine_lr_period_multiplier')

        self.xtrain_batches, self.ytrain_batches, \
        self.xvalid_batches, self.yvalid_batches, \
        self.xtest_batches, self.ytest_batches = \
            gdata.prepare_data(self.dataset, batch_size=self.batch_size, valid_size=self.valid_size,
                               return_test=True, batch_test=True)

        # self.macro_graph = mg.MacroGraph(input_shape=self.xtest_batches[0].shape, output_depths=self.output_depths,
        #                                  regularization_rate=5e-4)
        # self.macro_sess = self.macro_graph.sess
        # self.controller = ctrl.Controller(input_shape=self.xtrain_batches[0].shape,
        #                                   n_op_layers=len(self.output_depths),
        #                                   num_op_types=len(mg.OP_TYPES),
        #                                   initializer=self.initializer, **kwargs)
        # self.ctrl_sess = self.controller.sess


        # self.output_depths = kwargs.get('output_depths')
        # self.input_depths = kwargs.get('input_depths')
        self.input_depths = [36, 36, 36, 36, 36, 36]
        self.output_depths = [36, 36, 36, 36, 36, 36]
        # self.working_shape = kwargs.get('working_shapes')
        self.working_shapes = [[32, 32], [16, 16], [16, 16], [8, 8], [8, 8], [8, 8]]

    def build_subgraph(self, op_seq, arc_seq, input_shape):
        g = c.v1.Graph()
        with g.as_default():
            batch_acc_tuple = self.make_accurcy_metric(g, 'batch_acc')
            epoch_acc_tuple = self.make_accurcy_metric(g, 'epoch_acc')
            acc_metrics = (batch_acc_tuple, epoch_acc_tuple)

            with c.v1.variable_scope('g', reuse=c.v1.AUTO_REUSE):
                with c.v1.variable_scope('input'):
                    x = c.v1.placeholder(dtype=tf.float32, shape=input_shape, name='x')
                    y = c.v1.placeholder(dtype=tf.int32, shape=(input_shape[0],), name='y')
                    learning_rate = c.v1.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

                stem_filters = self.get_variable('filters', 'stem', shape=(3, 3, input_shape[-1],
                                                                           self.output_depths[0]))
                out = tf.nn.conv2d(x, stem_filters, strides=1, padding="SAME", data_format=NHWC)
                with c.v1.variable_scope(f'layer_0'):
                    out = self.apply_batchnorm(input_tensor=out, channels=self.output_depths[0], id='L0')
                    out = tf.nn.relu(out)
                    out = self.apply_layer(op_seq=op_seq, input_tensor=out, layer_index=0)
                    out = tf.reshape(out, shape=(self.batch_size, self.working_shapes[0][0], self.working_shapes[0][1],
                                                 self.output_depths[0]))
                    layer_inputs = [out]

                for layer_index in range(1, len(self.output_depths)):
                    with c.v1.variable_scope(f'layer_{layer_index}'):
                        out = layer_inputs[-1]
                        out = self.apply_batchnorm(out, self.input_depths[layer_index], id='pre')
                        out = tf.nn.relu(out)
                        out = self.apply_layer(op_seq, out, layer_index)
                        layer_inputs.append(out)

                        out = self.consolidate_residuals(arc_seq, layer_inputs, layer_index)
                        layer_inputs[-1] = out

                        if layer_index == 1 or layer_index == 3:
                            layer_inputs = self.factor_reduce(layer_inputs, layer_index)

                with c.v1.variable_scope('fc0'):
                    weights = c.v1.get_variable('weights', initializer=self.initializer,
                                                shape=(self.output_depths[-1], 10), dtype=tf.float32, trainable=True)
                    bias = c.v1.get_variable('bias',
                                             initializer=tf.constant_initializer(0.0), shape=[10],
                                             trainable=True)

                    out = layer_inputs[-1]
                    out = tf.reduce_mean(out, axis=[1, 2])
                    out = tf.nn.dropout(out, self.dropout_prob)
                    out = tf.reshape(out, shape=(self.batch_size, -1))
                    out = tf.matmul(out, weights)
                    # out = tf.matmul(out, weights2)
                    out = tf.add(out, bias)

                logits = tf.identity(out, name='logits')
                prediction = tf.nn.softmax(out, name='prediction')
                cross_entropy = tf.losses.sparse_categorical_crossentropy(y, prediction, from_logits=False)
                loss = tf.reduce_mean(cross_entropy, name='loss')

                squared_weights = [tf.square(g) for g in g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)]
                mean_squares = [tf.reduce_mean(sw) for sw in squared_weights]
                mean_reg_losses = tf.reduce_mean(mean_squares)

                loss += self.regularization_rate * mean_reg_losses

                gradients = c.v1.gradients([loss], g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))
                optimizer = self.optimizer(learning_rate=learning_rate)
                train_step = optimizer.apply_gradients(
                    zip(gradients, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)))
                init = c.v1.global_variables_initializer()

            return g, init, train_step, loss, logits, acc_metrics




    def apply_layer(self, op_seq, input_tensor, layer_index):
        op_type = op_seq[layer_index]

        if op_type == 0:
            return self.apply_conv2d(input_tensor=input_tensor,
                                     conv_shape=[3, 3, self.input_depths[layer_index], self.output_depths[layer_index]])
        elif op_type == 1:
            return self.apply_conv2d(input_tensor=input_tensor,
                                     conv_shape=[5, 5, self.input_depths[layer_index], self.output_depths[layer_index]])
        elif op_type == 2:
            return self.apply_sep_conv2d(input_tensor=input_tensor,
                                         conv_shape=[3, 3, self.input_depths[layer_index], 1],
                                         output_depth=self.output_depths[layer_index])
        elif op_type == 3:
            return self.apply_sep_conv2d(input_tensor=input_tensor,
                                         conv_shape=[5, 5, self.input_depths[layer_index], 1],
                                         output_depth=self.output_depths[layer_index])
        elif op_type == 4:
            return self.apply_pool(input_tensor=input_tensor, layer_index=layer_index, type=AVG)
        elif op_type == 5:
            return self.apply_pool(input_tensor=input_tensor, layer_index=layer_index, type=MAX)
        else:
            raise NotImplementedError()




    def apply_conv2d(self, input_tensor, conv_shape):
        with c.v1.variable_scope(f'{CONV}{conv_shape[0]}x{conv_shape[1]}'):
            pointwise_filters = self.get_variable('filters', 'pointwise', shape=(1, 1, conv_shape[2], conv_shape[3]))
            out = tf.nn.conv2d(input_tensor, pointwise_filters, strides=1, padding="SAME", data_format=NHWC)
            out = self.apply_batchnorm(input_tensor=out, channels=conv_shape[3], id='0')
            out = tf.nn.relu(out)
            conv_filters = self.get_variable('filters', 'conv',
                                             shape=(conv_shape[0], conv_shape[1], conv_shape[3], conv_shape[3]))
            out = tf.nn.conv2d(out, conv_filters, strides=1, padding="SAME", data_format=NHWC)
            return out

    def apply_sep_conv2d(self, input_tensor, conv_shape, output_depth):
        with c.v1.variable_scope(f'{SEP}{conv_shape[0]}x{conv_shape[1]}'):
            pointwise_0 = self.get_variable('pw_filters_0', f'{SEP}{conv_shape[0]}x{conv_shape[1]}',
                                            shape=(1, 1, conv_shape[2], output_depth))
            out = tf.nn.conv2d(input_tensor, pointwise_0, strides=1, padding="SAME", data_format=NHWC)
            out = self.apply_batchnorm(out, output_depth, '1')
            out = tf.nn.relu(out)

            depthwise_filters = self.get_variable("dw_filters", f'{SEP}{conv_shape[0]}x{conv_shape[1]}',
                                                  shape=(conv_shape[0], conv_shape[1], output_depth, 1))
            pointwise_1 = self.get_variable('pw_filters_1', f'{SEP}{conv_shape[0]}x{conv_shape[1]}',
                                            shape=(1, 1, output_depth, output_depth))
            out = tf.nn.separable_conv2d(out, depthwise_filters, pointwise_1,
                                         strides=[1, 1, 1, 1], padding="SAME", data_format=NHWC)
            return out

    def apply_pool(self, input_tensor, layer_index, type):
        with c.v1.variable_scope(f'{type}pool'):
            pointwise_filters = self.get_variable('pw_filters', f'{type}pool_{layer_index}', shape=(
                1, 1, self.input_depths[layer_index], self.output_depths[layer_index]))
            out = tf.nn.conv2d(input_tensor, pointwise_filters, strides=1, padding="SAME", data_format=NHWC)
            out = self.apply_batchnorm(out, self.output_depths[layer_index], 'pool')

            if type == AVG:
                out = tf.nn.avg_pool2d(out, ksize=3, strides=1, padding="SAME", data_format=NHWC, name=f'{type}pool')
            elif type == MAX:
                out = tf.nn.max_pool2d(out, ksize=3, strides=1, padding="SAME", data_format=NHWC, name=f'{type}pool')
            else:
                raise NotImplementedError()
            return out

    def consolidate_residuals(self, arc_seq, layer_inputs, layer_index):
        with c.v1.variable_scope(f'consolidate_L{layer_index}'):
            connection = arc_seq[layer_index]
            residuals = []
            for target_index in range(len(layer_inputs) - 1):
                if connection[target_index] == 1:
                    residuals.append(layer_inputs[target_index])
                else:
                    residuals.append(tf.zeros_like(layer_inputs[target_index]))
            residuals.append(layer_inputs[-1])
            out = tf.add_n(residuals)
            out = tf.reshape(out, shape=(self.batch_size, self.working_shapes[layer_index-1][0],
                                         self.working_shapes[layer_index-1][1], self.output_depths[layer_index]))
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
                pw_filters_path1 = c.v1.get_variable('pw_filtesr_path1',
                                                     initializer=self.initializer,
                                                     shape=shape,
                                                     dtype=tf.float32,
                                                     regularizer=None, trainable=True)
                shape = (1, 1, self.output_depths[target_index], self.output_depths[layer_index] // 2)
                pw_filters_path2 = c.v1.get_variable('pw_filters_path2',
                                                     initializer=self.initializer,
                                                     shape=shape,
                                                     dtype=tf.float32,
                                                     regularizer=None, trainable=True)
                path1 = tf.nn.avg_pool2d(input_tensor, ksize=1, strides=[1, 2, 2, 1], padding="VALID", data_format=NHWC)
                path1 = tf.nn.conv2d(path1, pw_filters_path1, strides=1, padding="SAME", data_format=NHWC)

                pads = [[0, 0], [0, 1], [0, 1], [0, 0]]
                path2 = tf.pad(input_tensor, pads)[:, 1:, 1:, :]
                path2 = tf.nn.avg_pool2d(path2, ksize=1, strides=[1, 2, 2, 1], padding="VALID", data_format=NHWC)
                path2 = tf.nn.conv2d(path2, pw_filters_path2, strides=1, padding="SAME", data_format=NHWC)

                out = tf.concat([path1, path2], axis=3)
                out = self.apply_batchnorm(out, self.output_depths[layer_index],
                                           id=f'fr_L{layer_index}_from_{target_index}')
                return out



    def get_variable(self, var_name, scope_name, shape, dtype=tf.float32, initialiser=None,
                     reuse=c.v1.AUTO_REUSE, regulariser=None, trainable=True):
        initialiser = self.initializer if initialiser is None else initialiser
        with c.v1.variable_scope(scope_name):
            return c.v1.get_variable(var_name, initializer=self.initializer, shape=shape,
                                     dtype=dtype, trainable=trainable)

    def apply_batchnorm(self, input_tensor, channels, id):
        with c.v1.variable_scope(f'batchnorm_{id}'):
            mu = c.v1.get_variable('mu', initializer=tf.constant_initializer(0.0), shape=[channels],
                                   trainable=False)
            sigma = c.v1.get_variable('sigma', initializer=tf.constant_initializer(1.0), shape=[channels],
                                      trainable=False)
            offset = c.v1.get_variable('offset', initializer=tf.constant_initializer(0.0), shape=[channels],
                                       trainable=True)
            scale = c.v1.get_variable('scale', initializer=tf.constant_initializer(1.0), shape=[channels],
                                      trainable=True)
        out = tf.nn.batch_normalization(input_tensor, mu, sigma, offset, scale, variance_epsilon=EPS)
        return out

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

    def update_accuracy(self, sess, acc_tuple, logits, labels):
        sess.run(acc_tuple[1], {acc_tuple[2]: logits, acc_tuple[3]: labels})

    def retrieve_accuracy(self, sess, acc_tuple):
        return sess.run(acc_tuple[0])

    def reset_accuracy(self, sess, acc_tuple):
        sess.run(acc_tuple[4])

    # def load_gnas_graphs(self):
    #     saver = c.v1.train.import_meta_graph(self.macro_graph_save_path + f'-{self.checkpoint_id}.meta')


    def sample_architecture(self, sample=None):
        run_log = pickle.load(open(self.run_dir + 'log.pkl', 'rb'))
        architectures = run_log['ctrl_batch_architectures'][self.checkpoint_id][:sample]
        accuracies = run_log['ctrl_batch_accs'][self.checkpoint_id][:sample]
        best_architecture = architectures[np.argmax(accuracies)]
        return best_architecture

    def create_log(self):
        log = {
            'total_elapsed_process_time': [],
            'total_elapsed_perf_time': [],
            'total_epochs': [],
            'batch_accuracy': [],
            'batch_loss': [],
            'batch_process_time': [],
            'batch_perf_time': [],
            'epoch_accuracy': [],
            'epoch_loss': [],
            'epoch_process_time': [],
            'epoch_loss_time': [],
            'epoch_learn_rate': [],
            'val_accuracy': [],
            'test_accuracy': []
        }
        return log

    def plot_acc(self):
        log = pickle.load(open(self.output_dir + self.log_save_path, 'rb'))
        batch_losses = log['batch_loss']
        batch_accs = log['batch_accuracy']
        losses = [np.array(loss) for loss in batch_losses]
        accs = [np.array(acc) for acc in batch_accs]
        losses = np.concatenate(losses[:5])
        accs = np.concatenate(accs[:5])
        loss_series = pd.Series(losses)
        acc_series = pd.Series(accs)
        ema = loss_series.ewm(span=250).mean()
        emacc = acc_series.ewm(span=250).mean()
        plt.figure()
        plt.plot(ema)
        # plt.ylim([2.0, 2.5])
        plt.savefig(self.output_dir + 'analysis' + f'loss_vs_learning_rate__rounds.png')
        plt.figure()
        plt.plot(emacc)
        # plt.ylim([2.0, 2.5])
        plt.savefig(self.output_dir + 'analysis' + f'acc_vs_learning_rate_rounds.png')
        print('test')


    def get_cosine_learn_rates(self, epochs):
        learn_rates = []

        for e in range(1, epochs):
            periods = [0, self.macro_cosine_lr_init_period]

            while e - 1 >= periods[-1]:
                periods.append(periods[-1] + (periods[-1] - periods[-2]) * self.macro_cosine_lr_period_multiplier)
            progress = (e - 1 - periods[-2]) / (periods[-1] - periods[-2])

            lr = self.macro_cosine_lr_min + \
                 ((self.macro_cosine_lr_max - self.macro_cosine_lr_min) / 2) * (1 + cos(pi * progress))
            learn_rates.append(lr)
        return learn_rates

    def get_exp_decay_learn_rates(self, start, decay, iters):
        learn_rates = [start]
        for i in range(1, iters):
            learn_rates.append(start * pow(e, decay * i))
        return learn_rates

    def eval_arc(self, architecture):
        start_prcs, start_perf = process_time(), perf_counter()
        with tf.device('/GPU:0'):
            xtrain, ytrain = self.xtrain_batches, self.ytrain_batches
            xval, yval = self.xvalid_batches, self.yvalid_batches
            xtest, ytest = self.xtest_batches, self.ytest_batches
        op_seq, arc_seq = architecture[0][0], architecture[0][1]
        g, init, train_step, loss, logits, acc_metrics = self.build_subgraph(op_seq, arc_seq,
                                                                             input_shape=xtrain[0].shape)
        log = self.create_log()
        epochs = 3000
        learn_rates_ = self.get_cosine_learn_rates(epochs)
        # learn_rates_ = self.get_exp_decay_learn_rates(start=0.000001, decay=0.001474, iters=7000)
        with tf.device('/GPU:0'):
            learn_rates = learn_rates_
        self.batch_acc = acc_metrics[0]
        self.epoch_acc = acc_metrics[1]
        iter = 0
        with c.v1.Session(graph=g) as sess:
            sess.run(init)
            sess.run(c.v1.local_variables_initializer())

            for e in range(epochs):
                epoch_prcs_start, epoch_perf_start = process_time(), perf_counter()
                self.reset_accuracy(sess, self.epoch_acc)
                start = time.time()
                np.random.RandomState(seed=e).shuffle(xtrain)
                np.random.RandomState(seed=e).shuffle(ytrain)

                batch_accs, batch_losses, batch_prcs_times, batch_perf_times = [], [], [], []

                for b in range(len(xtrain)):

                    batch_prcs_start, batch_perf_start = process_time(), perf_counter()
                    feed_dict = {'g/input/x:0': xtrain[b],
                                 'g/input/y:0': ytrain[b],
                                 'g/input/learning_rate:0': learn_rates[e]}
                    fetches = [train_step, logits, loss]
                    _, logits_out, batch_loss = sess.run(fetches, feed_dict)
                    self.update_accuracy(sess, self.batch_acc, logits_out, ytrain[b])
                    self.update_accuracy(sess, self.epoch_acc, logits_out, ytrain[b])
                    batch_acc = self.retrieve_accuracy(sess, self.batch_acc)
                    self.reset_accuracy(sess, self.batch_acc)
                    batch_accs.append(batch_acc)
                    batch_losses.append(batch_loss)
                    batch_prcs_times.append(process_time() - batch_prcs_start)
                    batch_perf_times.append(perf_counter() - batch_perf_start)
                    iter += 1

                    # if b % 100 == 0:
                    #     print(f'n-hundredth batch, rolling accuracy = {epoch_acc}, time={time.time() - batch_start}')
                epoch_acc = self.retrieve_accuracy(sess, self.epoch_acc)
                log['batch_accuracy'].append(batch_accs)
                log['batch_loss'].append(batch_losses)
                log['batch_process_time'].append(np.mean(batch_prcs_times))
                log['batch_perf_time'].append(np.mean(batch_perf_times))
                log['epoch_loss'].append(np.mean(batch_losses))
                log['epoch_accuracy'].append(epoch_acc)

                fetches = [loss, logits]
                self.reset_accuracy(sess, self.epoch_acc)
                for v in range(len(xval)):
                    loss_, logits_out_ = sess.run(fetches, {'g/input/x:0': xval[v], 'g/input/y:0': yval[v]})
                    self.update_accuracy(sess, self.epoch_acc, logits_out_, yval[v])

                val_accuracy = self.retrieve_accuracy(sess, self.epoch_acc)
                log['val_accuracy'].append(val_accuracy)

                self.reset_accuracy(sess, self.epoch_acc)
                for t in range(len(xtest)):
                    loss_, logits_out_ = sess.run(fetches, {'g/input/x:0': xtest[t], 'g/input/y:0': ytest[t]})
                    self.update_accuracy(sess, self.epoch_acc, logits_out_, ytest[t])

                test_acc = self.retrieve_accuracy(sess, self.epoch_acc)
                log['test_accuracy'].append(test_acc)

                pickle.dump(log, open(self.output_dir + self.log_save_path, 'wb'))

                print(f'epoch={e}, accuracy = {epoch_acc}, time={time.time() - start}, val accuracy = {val_accuracy}, test_accuracy = {test_acc}')


def eval_arc(**kwargs):
    ae = ArchitectureEval(**kwargs)
    architecture = ae.sample_architecture(sample=None)
    ae.eval_arc(architecture)
    # ae.plot_acc()



if __name__ == '__main__':
    params = {
        'output_dir': '../output/evals/46 /',
        'run_dir': '../output/47-nognn/',
        'checkpoint_id': 69,
        'dataset': 'cifar10',
        'valid_size': 0.2,
        'log_save_path': 'log.pkl',
        'initializer': tf.keras.initializers.GlorotUniform,
        'batch_size': 128,
        'dropout': 0.5,
        'regularization_rate': 6e-3,
        'optimizer': c.v1.train.AdamOptimizer,
        'macro_cosine_lr_init_period': 10,
        'macro_cosine_lr_period_multiplier': 2,
        'macro_cosine_lr_min': 0.000025,
        'macro_cosine_lr_max': 0.002500

    }
    if os.path.exists(params['output_dir']):
        raise ValueError('might overwrite')
        pass
    else:
        os.makedirs(params['output_dir'])
    eval_arc(**params)
