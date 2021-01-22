from tensorflow.keras.initializers import RandomUniform, GlorotUniform, HeNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow import compat as c
import tensorflow as tf
import numpy as np
###########################################################################
#################### choose controller version here #######################
###########################################################################
import controller_gnn as ctrl
# import controller_standard as ctrl
###########################################################################
###########################################################################
import macro_graph as mg
import gnas_data as gdata
import gnas_utils as gutil
import pickle
from time import process_time, perf_counter
from functools import partial
import os
import shutil
from math import cos, pi, e, pow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.debugging.set_log_device_placement(True)

class GNAS(object):
    def __init__(self, controller_units=100,
                 eps_per_ctrl_batch=30, fork=False, **kwargs):
        super(GNAS, self).__init__()
        self.output_dir = kwargs.get('output_dir')
        if not fork:
            if os.path.exists(self.output_dir):
                if not kwargs.get('testing'):
                    raise ValueError('Path already exists, might overwrite')
            # if not kwargs.get('testing'):
            os.makedirs(self.output_dir)
        pickle.dump(kwargs, open(self.output_dir + 'params.pkl', 'wb'))
        self.current_iteration = 1
        self.current_macro_batch = 0
        self.current_ctrl_batch = 0
        self.eps_per_ctrl_batch = kwargs.get('eps_per_ctrl_batch')
        self.ctrl_batches_per_iter = kwargs.get('ctrl_batches_per_iter')
        self.log_filename = kwargs.get('log_save_path')
        self.log_save_path = self.output_dir + self.log_filename
        self.macro_graph_filename = kwargs.get('macro_graph_save_path')
        self.macro_graph_save_path = self.output_dir + self.macro_graph_filename
        self.controller_graph_filename = kwargs.get('ctrl_graph_save_path')
        self.controller_graph_save_path = self.output_dir + self.controller_graph_filename
        self.dataset = kwargs.get('dataset')
        self.batch_size = kwargs.get('batch_size')
        self.output_depths = kwargs.get('output_depths')
        self.valid_size = kwargs.get('valid_size')
        self.macro_cosine_lr_max = kwargs.get('macro_cosine_lr_max')
        self.macro_cosine_lr_min = kwargs.get('macro_cosine_lr_min')
        self.macro_cosine_lr_init_period = kwargs.get('macro_cosine_lr_init_period')
        self.macro_cosine_lr_period_multiplier = kwargs.get('macro_cosine_lr_period_multiplier')
        self.macro_learning_rate = kwargs.get('macro_learning_rate')
        self.ctrl_learning_rate = kwargs.get('ctrl_learning_rate')
        self.max_saves_to_keep = kwargs.get('max_saves_to_keep')
        self.keep_checkpoint_every_n_hours = kwargs.get('keep_checkpoint_every_n_hours')
        if not os.path.exists(self.output_dir + self.macro_graph_save_path):
            os.makedirs(self.output_dir + self.macro_graph_save_path)
        if not os.path.exists(self.output_dir + self.controller_graph_save_path):
            os.makedirs(self.output_dir + self.controller_graph_save_path)
        self.output_depths = [16, 16, 32, 32, 64, 128, 256, 512] if self.output_depths is None else self.output_depths
        self.controller_units = controller_units
        self.initializer = RandomUniform(-0.1, 0.1)

        self.xtrain_batches, self.ytrain_batches, \
        self.xvalid_batches, self.yvalid_batches, \
        self.xtest_batches, self.ytest_batches = \
            gdata.prepare_data(self.dataset, batch_size=self.batch_size, valid_size=self.valid_size,
                               return_test=True, batch_test=True)

        self.ctrl_op_type_entropy_weight = kwargs.get('ctrl_op_type_entropy_weight')
        self.ctrl_connection_entropy_weight = kwargs.get('ctrl_connection_entropy_weight')
        self.ctrl_skip_penalty_weight = kwargs.get('ctrl_skip_penalty_weight')

        self.controller = ctrl.Controller(input_shape=self.xtrain_batches[0].shape,
                                          n_op_layers=len(self.output_depths),
                                          num_op_types=len(mg.OP_TYPES),
                                          initializer=self.initializer, **kwargs)
        self.ctrl_sess = self.controller.sess
        self.ctrl_saver = c.v1.train.Saver(self.ctrl_sess.graph.get_collection(c.v1.GraphKeys.GLOBAL_VARIABLES),
                                           max_to_keep=self.max_saves_to_keep,
                                           keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)
        self.macro_graph = mg.MacroGraph(input_shape=self.xtrain_batches[0].shape, output_depths=self.output_depths,
                                         regularization_rate=5e-4)
        self.macro_sess = self.macro_graph.sess
        self.macro_saver = c.v1.train.Saver(self.macro_sess.graph.get_collection(c.v1.GraphKeys.GLOBAL_VARIABLES),
                                            max_to_keep=self.max_saves_to_keep,
                                            keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

        # self.ctrl_optimizer = Adam(learning_rate=controller_lr)
        self.eps_per_ctrl_batch = eps_per_ctrl_batch

        self.macro_loss_fn = SparseCategoricalCrossentropy()
        self.accuracy_fn = SparseCategoricalAccuracy()
        self.epoch_accuracy_fn = SparseCategoricalAccuracy()

        self.macro_lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9,
                                                  staircase=True)
        self.macro_optimizer = Adam(learning_rate=self.macro_lr_schedule)

        self.macro_moving_avg = 1 / (max(self.ytrain_batches[0]) + 1)
        self.ctrl_moving_avg = 1 / (max(self.ytrain_batches[0]) + 1)
        self.macro_exp_ma_alpha = 2 / ((len(self.xtrain_batches) / 2) + 1)
        self.ctrl_exp_ma_alpha = 2 / 201
        self.ctrl_reward_mv_avg = 0.
        self.ctrl_reward_exp_ma_alpha = 2 / 201

        self.op_type_entropy_weight = 0.1
        self.connection_entropy_weight = 0.1
        self.skip_penalty_weight = 0.8

        self.round_log = {
            'macro_total_train_steps': 0,
            'macro_batch_architectures': [],
            'macro_batch_losses': [],
            'macro_batch_accs': [],
            'macro_moving_averages': [],
            'macro_batch_eval_process_time': [],
            'macro_batch_eval_perf_time': [],
            'macro_batch_learn_rates': [],
            'macro_iteration_reported_acc': 0,
            'macro_iteration_eval_process_time': 0,
            'macro_iteration_eval_perf_time': 0,
            'macro_iteration_learn_rate': 0,
            'ctrl_total_train_steps': 0,
            'ctrl_batch_architectures': [],
            'ctrl_batch_losses': [],
            'ctrl_batch_accs': [],
            'ctrl_moving_averages': [],
            'ctrl_reward_mv_avg': [],
            'ctrl_batch_process_time': [],
            'ctrl_batch_perf_time': [],
            'ctrl_iteration_reported_acc': 0,
            'ctrl_iteration_process_time': 0,
            'ctrl_iteration_perf_time': 0
        }

    def macro_cosine_lr(self, **kwargs):
        periods = [0, self.macro_cosine_lr_init_period]

        while self.current_iteration-1 >= periods[-1]:
            periods.append(periods[-1] + (periods[-1] - periods[-2])*self.macro_cosine_lr_period_multiplier)
        progress = (self.current_iteration-1 - periods[-2]) / (periods[-1] - periods[-2])

        lr = self.macro_cosine_lr_min + \
             ((self.macro_cosine_lr_max - self.macro_cosine_lr_min)/2) * (1 + cos(pi * progress))
        return lr

    def macro_batch_exp_increase_lr(self, batch, max_batch=1250, **kwargs):
        if batch is None:
            return 0
        n = (self.current_iteration-1) * max_batch
        n += batch
        # exponential increase from 0.000001 to 5.0 over 2 epochs (2500 batches)
        # return 0.000001 * pow(e, 0.00617 * n)
        # exponential increase from 0.0001 to 5.0 over 2 epochs (2500 batches)
        # return 0.0001 * pow(e, 0.00433 * n)
        # exponential increase from 0.0000001 to 5.0 over 4 epochs (5000 batches)
        # return 0.0000001 * pow(e, 0.0035455 * n)
        # exponential increase from 0.00001 to 1.0 over 4 epochs (5000 batches)
        return 0.00001 * pow(e, 0.002303 * n)

    def ctrl_batch_exp_decay(self, **kwargs):
        n = (self.current_iteration-1) * self.ctrl_batches_per_iter
        n += self.current_ctrl_batch
        # exponential increase from 0.000001 to 5.0 over 360 ctrl batches
        # return 0.000001 * pow(e, 0.04285 * n)
        # exponential increase from 0.000001 to 5.0 over 720 ctrl batches
        # return 0.000001 * pow(e, 0.02142 * n)
        # exponential increase from 0.0000001 to 5.0 over 1440 ctrl batches
        # return 0.0000001 * pow(e, 0.01231 * n)
        # exponential increase from 0.0000001 to 5.0 over 720 ctrl batches
        # return 0.0000001 * pow(e, 0.02462 * n)
        # exponential increase from 0.000001 to 1.0 over 1440 ctrl batches
        # return 0.000001 * pow(e, 0.009594 * n)
        # exponential increase from 0.000001 to 1.0 over 2000 ctrl batches
        # return 0.000001 * pow(e, 0.00691 * n)
        # exponential increase from 0.00000001 to 1.0 over 4000 ctrl batches
        # return 0.00000001 * pow(e, 0.00460517 * n)
        # exponential increase from 0.000001 to 0.05 over 10000 ctrl batches
        return 0.000001 * pow(e, 0.00108197 * n)

    def ctrl_fixed_lr(self, **kwargs):
        if self.current_iteration < 0:
            return 0
        else:
            return kwargs.get('learning_rate')


    def restore_macro_graph(self, file_id):
        saver = c.v1.train.import_meta_graph(self.macro_graph_save_path + f'-{file_id}.meta')
        # saver.restore(self.macro_sess, self.macro_graph_save_path + f'-{file_id}.data-00000-of-00001')
        # saver.restore(self.macro_sess, self.macro_graph_save_path + f'model.ckpt-{file_id}')
        saver.restore(self.macro_sess, self.macro_graph_save_path + f'-{file_id}')

    def restore_controller_graph(self, file_id):
        saver = c.v1.train.import_meta_graph(self.controller_graph_save_path + f'-{file_id}.meta')
        # saver.restore(self.ctrl_sess, self.controller_graph_save_path + f'-{file_id}.data-00000-of-00001')
        saver.restore(self.ctrl_sess, self.controller_graph_save_path + f'-{file_id}')

    def fork_directory(self, fork_dir):
        if os.path.exists(fork_dir):
            raise ValueError("Directory already exists")
        shutil.copytree(self.output_dir, fork_dir)
        self.output_dir = fork_dir
        self.log_save_path = self.output_dir + self.log_filename
        self.macro_graph_save_path = self.output_dir + self.macro_graph_filename
        self.controller_graph_save_path = self.output_dir + self.controller_graph_filename
        if not os.path.exists(self.macro_graph_save_path):
            os.makedirs(self.macro_graph_save_path)
        if not os.path.exists(self.controller_graph_save_path):
            os.makedirs(self.controller_graph_save_path)




    def sample_valid_architecture(self):
        while True:
            # tf.config.run_functions_eagerly(True)
            # tf.config.experimental_run_functions_eagerly(True)
            # tf.config.experimental_functions_run_eagerly()
            controller_fetches = [self.controller.op_seq, self.controller.arc_seq]
            controller_feed_dict = self.controller.create_feed_dict(training=False)
            op_types, connect_seq = self.ctrl_sess.run(controller_fetches, controller_feed_dict)
            op_types = np.array(op_types)
            # if self.macro_graph.validate_op_types(op_types):
            return op_types, connect_seq

    def train_macro_batch(self, x_batch, y_batch, **kwargs):
        op_types, connections = self.sample_valid_architecture()
        # print(op_types)
        # print(connections)
        learn_rate = self.macro_learning_rate(self, **kwargs)
        feed_dict = self.macro_graph.create_feed_dict(x_batch, y_batch, op_types, connections, learn_rate)
        fetches = [self.macro_graph.train_step, self.macro_graph.loss, self.macro_graph.logits]
        _, loss, logits_out = self.macro_sess.run(fetches, feed_dict)

        self.macro_graph.reset_accuracy(self.macro_graph.macro_batch_acc)
        self.macro_graph.update_accuracy(self.macro_graph.macro_batch_acc, logits_out, y_batch)
        self.macro_graph.update_accuracy(self.macro_graph.macro_epoch_acc, logits_out, y_batch)
        acc = self.macro_graph.retrieve_accuracy(self.macro_graph.macro_batch_acc)

        self.macro_moving_avg = (self.macro_exp_ma_alpha * acc) + (1 - self.macro_exp_ma_alpha) * self.macro_moving_avg

        return acc, loss, op_types, connections, learn_rate

    def train_macro_graph(self, iter, epochs=1, report_epoch=True, show_progress=True, batches_per_epoch=None):
        start_prcs, start_perf = process_time(), perf_counter()
        batches_per_epoch = len(self.xtrain_batches) if batches_per_epoch is None else batches_per_epoch

        for epoch in range(epochs):
            self.macro_graph.reset_accuracy(self.macro_graph.macro_epoch_acc)

            if show_progress:
                gutil.printProgressBar(0, batches_per_epoch, prefix='Macro round epoch progress',
                                       suffix='Complete', length=50)

            np.random.RandomState(seed=epoch).shuffle(self.xtrain_batches)
            np.random.RandomState(seed=epoch).shuffle(self.ytrain_batches)

            for b in range(batches_per_epoch):
                start_prcs, start_perf = process_time(), perf_counter()
                self.current_macro_batch = b
                batch_accuracy, batch_loss, op_types, connections, learn_rate = \
                    self.train_macro_batch(self.xtrain_batches[b], self.ytrain_batches[b], batch=b)
                self.round_log['macro_total_train_steps'] += 1
                self.round_log['macro_batch_architectures'].append((op_types, connections))
                self.round_log['macro_batch_accs'].append(batch_accuracy)
                self.round_log['macro_batch_losses'].append(batch_loss)
                self.round_log['macro_moving_averages'].append(self.macro_moving_avg)
                self.round_log['macro_batch_learn_rates'].append(learn_rate)
                self.round_log['macro_batch_eval_process_time'].append(process_time() - start_prcs)
                self.round_log['macro_batch_eval_perf_time'].append(perf_counter() - start_perf)
                self.round_log['macro_iteration_learn_rate'] = learn_rate
                if show_progress:
                    epoch_accuracy = self.macro_graph.retrieve_accuracy(self.macro_graph.macro_epoch_acc)
                    gutil.printProgressBar(b % batches_per_epoch + 1, batches_per_epoch,
                                           prefix='Macro round epoch progress',
                                           suffix=f'Complete. Train accuracy = {epoch_accuracy}', length=50)
        self.round_log['macro_iteration_reported_acc'] = \
            self.macro_graph.retrieve_accuracy(self.macro_graph.macro_epoch_acc)
        self.round_log['macro_iteration_eval_process_time'] += process_time() - start_prcs
        self.round_log['macro_iteration_eval_perf_time'] += perf_counter() - start_perf

    def generate_state_action_reward(self, data_batch):
        x_batch, y_batch = self.xvalid_batches[data_batch], self.yvalid_batches[data_batch]
        self.macro_graph.reset_accuracy(self.macro_graph.ctrl_batch_acc)
        feed_dict = self.controller.create_feed_dict(training=False)
        fetches = [self.controller.op_seq, self.controller.arc_seq, self.controller.list_of_things]
        op_seq_out, arc_seq_out, gnnout = self.ctrl_sess.run(fetches, feed_dict)
        if not self.macro_graph.validate_op_types(op_seq_out):
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_batch_acc,
                                             np.ones((128, 10), dtype=np.float32), -np.ones((128,), dtype=np.int32))
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_epoch_acc,
                                             np.ones((128, 10), dtype=np.float32), -np.ones((128,), dtype=np.int32))
        else:
            feed_dict = self.macro_graph.create_feed_dict(x_batch, y_batch, op_seq_out, arc_seq_out,
                                                          self.macro_learning_rate(self))
            fetches = [self.macro_graph.logits]
            logits_out = self.macro_sess.run(fetches, feed_dict)[0]
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_batch_acc, logits_out, y_batch)
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_epoch_acc, logits_out, y_batch)

        acc = self.macro_graph.retrieve_accuracy(self.macro_graph.ctrl_batch_acc)

        return op_seq_out, arc_seq_out, acc

    def record_ctrl_batch(self, data_batch):
        op_seqs = []
        arc_seqs = []
        accuracys = []

        for episode in range(self.eps_per_ctrl_batch):
            if data_batch % len(self.xvalid_batches) == 0:
                gdata.shuffle_xy(self.xvalid_batches, self.yvalid_batches)
                data_batch = 0

            op_seq_out, arc_seq_out, acc = self.generate_state_action_reward(data_batch)
            op_seqs.append(op_seq_out)
            arc_seqs.append(arc_seq_out)
            accuracys.append(acc)

            self.ctrl_moving_avg = (self.ctrl_exp_ma_alpha * acc) + (1 - self.ctrl_exp_ma_alpha) * self.ctrl_moving_avg

            data_batch += 1

        return op_seqs, arc_seqs, accuracys, data_batch

    def reinforce_from_SARs(self, op_seqs, arc_seqs, accuracies):
        trainable_vars = self.ctrl_sess.graph.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)
        # ctrl_batch_grads = [np.zeros(trainable_vars[i].shape) for i in range(len(trainable_vars))]
        losses = []
        for episode in range(self.eps_per_ctrl_batch):
            feed_dict = self.controller.create_feed_dict(training=True,
                                                         prev_accuracy=accuracies[episode],
                                                         moving_average=self.ctrl_reward_mv_avg,
                                                         prev_op_seq=op_seqs[episode],
                                                         prev_arc_seqs=arc_seqs[episode],
                                                         learning_rate=self.ctrl_learning_rate(self))
            gradients, loss, reward, gnnout, ag = self.ctrl_sess.run([self.controller.compute_gradients, self.controller.loss_fn,
                                                              self.controller.reward_fn,
                                                              self.controller.list_of_things, self.controller.apply_gradients],
                                                             feed_dict)
            self.ctrl_reward_mv_avg = (self.ctrl_reward_exp_ma_alpha * reward) + \
                                      (1 - self.ctrl_reward_exp_ma_alpha) * self.ctrl_reward_mv_avg
            # ctrl_batch_grads = [np.add(g1, g2) for g1, g2 in zip(ctrl_batch_grads, gradients)]
            losses.append(loss)


        # feed_dict = self.controller.create_feed_dict(training=False, gradients=ctrl_batch_grads)
        # self.controller.apply_gradients(feed_dict=feed_dict)
        # self.ctrl_sess.run(self.controller.apply_gradients, feed_dict)
        self.round_log['ctrl_total_train_steps'] += 1
        # print(f', sample op seq = {op_seqs[-1]}={sum(op_seqs[-1])} (loss={losses[-1]})')
        return losses

    def train_controller(self, iter, ctrl_batches=20, report_batch=True, show_progress=True):
        start_iter_prcs, start_iter_perf = process_time(), perf_counter()
        self.macro_graph.reset_accuracy(self.macro_graph.ctrl_epoch_acc)

        data_batch = 0
        if show_progress:
            gutil.printProgressBar(0, self.ctrl_batches_per_iter, prefix='Controller round progress ', suffix='Complete', length=50)

        for ctrl_batch in range(self.ctrl_batches_per_iter):
            start_prcs, start_perf = process_time(), perf_counter()
            self.macro_graph.reset_accuracy(self.macro_graph.ctrl_batch_acc)
            self.current_ctrl_batch = ctrl_batch
            self.current_macro_batch = ctrl_batch
            op_seqs, arc_seqs, accuracies, data_batch = self.record_ctrl_batch(data_batch)
            losses = self.reinforce_from_SARs(op_seqs, arc_seqs, accuracies)
            self.round_log['ctrl_total_train_steps'] += len(op_seqs)
            architectures = [(ops, arcs) for ops, arcs in zip(op_seqs, arc_seqs)]
            self.round_log['ctrl_batch_architectures'].append(architectures)
            self.round_log['ctrl_batch_losses'].append(losses)
            self.round_log['ctrl_moving_averages'].append(self.ctrl_moving_avg)
            self.round_log['ctrl_reward_mv_avg'].append(self.ctrl_reward_mv_avg)
            self.round_log['ctrl_batch_accs'].append(accuracies)
            self.round_log['ctrl_batch_process_time'].append(process_time() - start_prcs)
            self.round_log['ctrl_batch_perf_time'].append(perf_counter() - start_perf)
            if show_progress:
                acc = self.macro_graph.retrieve_accuracy(self.macro_graph.ctrl_epoch_acc)
                gutil.printProgressBar(ctrl_batch % self.ctrl_batches_per_iter + 1, self.ctrl_batches_per_iter,
                                       prefix='Controller round progress ', suffix=f'Complete. Valid accuracy = {acc}',
                                       length=50)
        self.round_log['ctrl_iteration_reported_acc'] = \
            self.macro_graph.retrieve_accuracy(self.macro_graph.ctrl_epoch_acc)
        self.round_log['ctrl_iteration_process_time'] += process_time() - start_iter_prcs
        self.round_log['ctrl_iteration_perf_time'] += perf_counter() - start_iter_perf

    def continue_gnas(self, from_checkpoint_id, fork_id, iterations=10000, start_iter_override=None):
        fork_dir = self.output_dir[:-1] + f'-{fork_id}/'
        gnas.fork_directory(fork_dir)
        gnas.restore_macro_graph(from_checkpoint_id)
        gnas.restore_controller_graph(from_checkpoint_id)

        self.current_iteration = from_checkpoint_id if start_iter_override is None else start_iter_override
        log = pickle.load(open(self.log_save_path, 'rb'))
        for k, v in log.items():
            log[k] = log[k][:from_checkpoint_id]
        pickle.dump(log, open(self.log_save_path, 'wb'))
        del log

        for i in range(self.current_iteration, iterations):
            start_prcs, start_perf = process_time(), perf_counter()
            print(f'Round {i}.')
            self.train_macro_graph(iter=i, epochs=1, report_epoch=True, batches_per_epoch=None)
            self.train_controller(iter=i, report_batch=True)
            for s in range(1):
                self.sample_and_test()
            if i % 1 == 0:
                self.macro_saver.save(self.macro_sess, self.macro_graph_save_path, global_step=i)
                self.ctrl_saver.save(self.ctrl_sess, self.controller_graph_save_path, global_step=i)

            self.update_log(start_prcs, start_perf)
            self.current_iteration += 1

    def sample_and_test(self):
        self.macro_graph.reset_accuracy(self.macro_graph.ctrl_batch_acc)

        feed_dict = self.controller.create_feed_dict(training=False)
        fetches = [self.controller.op_seq, self.controller.arc_seq, self.controller.list_of_things]
        op_seq_out, arc_seq_out, gnnout = self.ctrl_sess.run(fetches, feed_dict)
        x, y = gdata.shuffle_xy(self.xtest_batches, self.ytest_batches)
        for b in range(len(x)):
            feed_dict = self.macro_graph.create_feed_dict(x[b], y[b], op_seq_out, arc_seq_out,
                                                          self.macro_learning_rate(self, batch=None))
            fetches = [self.macro_graph.logits]
            logits_out = self.macro_sess.run(fetches, feed_dict)[0]
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_batch_acc, logits_out, y[b])

        acc = self.macro_graph.retrieve_accuracy(self.macro_graph.ctrl_batch_acc)

        print(f'Sampled architecture:')
        print(op_seq_out)
        for i in range(1, len(arc_seq_out)):
            print(arc_seq_out[i])
        print(f'Full test set accuracy = {round(acc, 4)}')

    def execute_gnas(self, iterations=10000):
        self.macro_sess.run(self.macro_graph.initialise)
        self.macro_saver.save(self.macro_sess, self.macro_graph_save_path, global_step=0)
        self.ctrl_sess.run(self.controller.initialise)
        self.ctrl_saver.save(self.ctrl_sess, self.controller_graph_save_path, global_step=0)
        self.create_log()

        for i in range(1, iterations):
            start_prcs, start_perf = process_time(), perf_counter()
            print(f'Round {i}.')
            self.train_macro_graph(iter=i, epochs=1, report_epoch=True, batches_per_epoch=None)
            self.train_controller(iter=i, report_batch=True)
            for s in range(1):
                self.sample_and_test()
            if i % 1 == 0:
                self.macro_saver.save(self.macro_sess, self.macro_graph_save_path, global_step=i)
                self.ctrl_saver.save(self.ctrl_sess, self.controller_graph_save_path, global_step=i)

            self.update_log(start_prcs, start_perf)
            self.current_iteration += 1

        self.macro_sess.close()
        self.ctrl_sess.close()

    def create_log(self):
        log = {
            'total_elapsed_process_time': [],
            'total_elapsed_perf_time': [],
            'total_iterations': [],
            'macro_total_train_steps': [],
            'ctrl_total_train_steps': [],
            'iteration_eval_process_time': [],
            'iteration_eval_perf_time': [],
            'macro_batch_architectures': [],
            'macro_batch_accs': [],
            'macro_batch_eval_process_time': [],
            'macro_batch_eval_perf_time': [],
            'macro_batch_losses': [],
            'macro_moving_averages': [],
            'macro_batch_learn_rates': [],
            'macro_iteration_reported_acc': [],
            'macro_iteration_eval_process_time': [],
            'macro_iteration_eval_perf_time': [],
            'macro_iteration_learn_rate': [],
            'ctrl_batch_architectures': [],
            'ctrl_batch_accs': [],
            'ctrl_batch_losses': [],
            'ctrl_moving_averages': [],
            'ctrl_reward_mv_avg': [],
            'ctrl_batch_process_time': [],
            'ctrl_batch_perf_time': [],
            'ctrl_iteration_reported_acc': [],
            'ctrl_iteration_process_time': [],
            'ctrl_iteration_perf_time': [],
        }
        pickle.dump(log, open(self.log_save_path, 'wb'))
        del log

    def update_log(self, start_prcs, start_perf):
        log = pickle.load(open(self.log_save_path, 'rb'))
        if log['total_elapsed_process_time'] == []:
            log['total_elapsed_process_time'] = [process_time() - start_prcs]
            log['total_elapsed_perf_time'] = [perf_counter() - start_perf]
            log['macro_total_train_steps'] = [self.round_log['macro_total_train_steps']]
            log['ctrl_total_train_steps'] = [self.round_log['ctrl_total_train_steps']]
        else:
            log['total_elapsed_process_time'].append(
                log['total_elapsed_process_time'][-1] + (process_time() - start_prcs))
            log['total_elapsed_perf_time'].append(
                log['total_elapsed_perf_time'][-1] + (perf_counter() - start_perf))
            log['macro_total_train_steps'].append(
                log['macro_total_train_steps'][-1] + self.round_log['macro_total_train_steps'])
            log['ctrl_total_train_steps'].append(
                log['ctrl_total_train_steps'][-1] + self.round_log['ctrl_total_train_steps'])
        log['total_iterations'].append(self.current_iteration)

        log['iteration_eval_process_time'].append(process_time() - start_prcs)
        log['iteration_eval_perf_time'].append(perf_counter() - start_perf)
        # log['macro_batch_architectures'].append(self.round_log['macro_batch_architectures'])
        log['macro_batch_accs'].append(self.round_log['macro_batch_accs'])
        log['macro_batch_eval_process_time'].append(np.mean(self.round_log['macro_batch_eval_process_time']))
        log['macro_batch_eval_perf_time'].append(np.mean(self.round_log['macro_batch_eval_perf_time']))
        log['macro_batch_losses'].append(self.round_log['macro_batch_losses'])
        log['macro_moving_averages'].append(np.mean(self.round_log['macro_moving_averages']))
        log['macro_batch_learn_rates'].append(np.mean(self.round_log['macro_batch_learn_rates']))
        log['macro_iteration_reported_acc'].append(self.round_log['macro_iteration_reported_acc'])
        log['macro_iteration_eval_process_time'].append(self.round_log['macro_iteration_eval_process_time'])
        log['macro_iteration_eval_perf_time'].append(self.round_log['macro_iteration_eval_perf_time'])
        log['macro_iteration_learn_rate'].append(self.round_log['macro_iteration_learn_rate'])
        log['ctrl_batch_architectures'].append(self.round_log['ctrl_batch_architectures'])
        log['ctrl_batch_accs'].append(self.round_log['ctrl_batch_accs'])
        log['ctrl_batch_losses'].append(np.mean(self.round_log['ctrl_batch_losses']))
        log['ctrl_moving_averages'].append(np.mean(self.round_log['ctrl_moving_averages']))
        log['ctrl_reward_mv_avg'].append(np.mean(self.round_log['ctrl_reward_mv_avg']))
        log['ctrl_batch_process_time'].append(np.mean(self.round_log['ctrl_batch_process_time']))
        log['ctrl_batch_perf_time'].append(np.mean(self.round_log['ctrl_batch_perf_time']))
        log['ctrl_iteration_reported_acc'].append(self.round_log['ctrl_iteration_reported_acc'])
        log['ctrl_iteration_process_time'].append(self.round_log['ctrl_iteration_process_time'])
        log['ctrl_iteration_perf_time'].append(self.round_log['ctrl_iteration_perf_time'])
        pickle.dump(log, open(self.log_save_path, 'wb'))
        del log
        self.round_log = {
            'macro_total_train_steps': 0,
            'macro_batch_architectures': [],
            'macro_batch_accs': [],
            'macro_batch_eval_process_time': [],
            'macro_batch_eval_perf_time': [],
            'macro_batch_losses': [],
            'macro_moving_averages': [],
            'macro_batch_learn_rates': [],
            'macro_iteration_reported_acc': 0,
            'macro_iteration_eval_process_time': 0,
            'macro_iteration_eval_perf_time': 0,
            'macro_iteration_learn_rate': 0,
            'ctrl_total_train_steps': 0,
            'ctrl_batch_architectures': [],
            'ctrl_batch_accs': [],
            'ctrl_batch_losses': [],
            'ctrl_moving_averages': [],
            'ctrl_reward_mv_avg': [],
            'ctrl_batch_process_time': [],
            'ctrl_batch_perf_time': [],
            'ctrl_iteration_reported_acc': 0,
            'ctrl_iteration_process_time': 0,
            'ctrl_iteration_perf_time': 0
        }


if __name__ == '__main__':
    # output_depths = [16, 16, 32, 32, 64, 128, 256, 512]
    # output_depths = [16, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512]
    # output_depths = [8, 16, 32, 64, 128, 256]
    # output_depths = [16, 16, 32, 32, 64, 64]

    params = {
        'output_dir': '../output/48-gnn/',
        'testing': False,
        'log_save_path': 'log.pkl',
        'dataset': 'cifar10',
        'valid_size': 0.2,
        'batch_size': 128,
        'height_dim': 1,
        'width_dim': 2,
        'macro_graph_save_path': 'macro_graph/model.ckpt',
        'ctrl_graph_save_path': 'controller_graph/model.ckpt',
        'max_saves_to_keep': 500,
        'keep_checkpoint_every_n_hours': 0.1,
        'output_depths': [36] * 6,
        'pool_layers': (3, 7),
        'pool_size': (2, 2),
        # 'ctrl_lr': 0.00035,
        # 'ctrl_learning_rate': GNAS.ctrl_batch_exp_decay,
        'ctrl_lr': 0.0004,
        'ctrl_learning_rate': partial(GNAS.ctrl_fixed_lr, learning_rate=0.0004),
        'ctrl_units': 100,
        'ctrl_initializer': RandomUniform(-0.1, 0.1),
        'ctrl_lstm_layers': 2,
        'ctrl_gnn_layer_dims': [10, 20, 40],
        'ctrl_num_op_types_override': None,
        'ctrl_skip_target': 0.4,
        'ctrl_tanh_constant': 1.5,
        'ctrl_temp': 5.0,
        'ctrl_optimizer': c.v1.train.AdamOptimizer,
        'ctrl_op_type_entropy_weight': 1.8,
        'ctrl_connection_entropy_weight': 3.0,
        'ctrl_skip_penalty_weight': 1.8,  # 0.8?
        'ctrl_batches_per_iter': 100,   # currently ~1m20s macro for 1500. 2000 -> 1m58s for ctrl.
        'eps_per_ctrl_batch': 1,
        'macro_regularization_rate': 5e-4,
        'macro_dropout_prob': 0.1,
        'macro_initializer': GlorotUniform,
        'macro_optimizer': c.v1.train.AdamOptimizer,
        'macro_learning_rate': GNAS.macro_cosine_lr,
        # 'macro_learning_rate': GNAS.macro_batch_exp_increase_lr,
        'macro_cosine_lr_max': 0.0015,
        'macro_cosine_lr_min': 0.000015,
        'macro_cosine_lr_init_period': 10,
        'macro_cosine_lr_period_multiplier': 2

    }

    # output_depths = [36] * 6
    # batch_size = 32
    c.v1.disable_eager_execution()
    gnas = GNAS(**params)
    gnas.execute_gnas(10000)
    # gnas = GNAS(fork=True, **params)
    # gnas.continue_gnas(from_checkpoint_id=9, fork_id=11, iterations=10000, start_iter_override=1)
