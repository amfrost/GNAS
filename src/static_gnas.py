from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow import compat as c
import tensorflow as tf
import numpy as np
import static_controller as ctrl
import more_newerer_macro_graph as mg
import gnas_data as gdata
import gnas_utils as gutil
import time

class GNAS(object):
    def __init__(self, dataset, output_depths, batch_size=128, controller_units=100, controller_lr=0.0035,
                 eps_per_ctrl_batch=30):
        super(GNAS, self).__init__()
        self.output_depths = [16, 16, 32, 32, 64, 128, 256, 512] if output_depths is None else output_depths
        self.controller_units = controller_units
        self.initializer = RandomUniform(-0.1, 0.1)

        self.xtrain_batches, self.ytrain_batches, self.xvalid_batches, self.yvalid_batches = \
            gdata.prepare_data(dataset, batch_size=batch_size, valid_size=0.2)

        self.controller = ctrl.Controller(units=controller_units,
                                          num_layers=len(self.output_depths),
                                          num_op_types=len(mg.OP_TYPES),
                                          initializer=self.initializer)
        self.ctrl_sess = self.controller.sess
        self.macro_graph = mg.MacroGraph(input_shape=self.xtrain_batches[0].shape, output_depths=self.output_depths,
                                         regularization_rate=5e-4)
        self.macro_sess = self.macro_graph.sess

        self.ctrl_optimizer = Adam(learning_rate=controller_lr)
        self.eps_per_ctrl_batch = eps_per_ctrl_batch

        self.macro_loss_fn = SparseCategoricalCrossentropy()
        self.accuracy_fn = SparseCategoricalAccuracy()
        self.epoch_accuracy_fn = SparseCategoricalAccuracy()

        self.macro_lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9,
                                                  staircase=True)
        self.macro_optimizer = Adam(learning_rate=self.macro_lr_schedule)

        self.moving_avg = 1 / (max(self.ytrain_batches[0]) + 1)
        self.exp_ma_alpha = 2 / ((len(self.xtrain_batches) / 2) + 1)

        self.op_type_entropy_weight = 0.1
        self.connection_entropy_weight = 0.1
        self.skip_penalty_weight = 0.8

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

    def train_macro_batch(self, x_batch, y_batch):
        op_types, connections = self.sample_valid_architecture()
        # print(op_types)
        # print(connections)
        feed_dict = self.macro_graph.create_feed_dict(x_batch, y_batch, op_types, connections)
        fetches = [self.macro_graph.train_step, self.macro_graph.logits]
        _, logits_out = self.macro_sess.run(fetches, feed_dict)

        self.macro_graph.reset_accuracy(self.macro_graph.macro_batch_acc)
        self.macro_graph.update_accuracy(self.macro_graph.macro_batch_acc, logits_out, y_batch)
        self.macro_graph.update_accuracy(self.macro_graph.macro_epoch_acc, logits_out, y_batch)
        acc = self.macro_graph.retrieve_accuracy(self.macro_graph.macro_batch_acc)

        self.moving_avg = (self.exp_ma_alpha * acc) + (1 - self.exp_ma_alpha) * self.moving_avg

        return acc, op_types, connections

    def train_macro_graph(self, iter, epochs=1, report_epoch=True, show_progress=True, batches_per_epoch=None):
        batches_per_epoch = len(self.xtrain_batches) if batches_per_epoch is None else batches_per_epoch

        for epoch in range(epochs):
            self.macro_graph.reset_accuracy(self.macro_graph.macro_epoch_acc)

            if show_progress:
                gutil.printProgressBar(0, batches_per_epoch, prefix='Macro round epoch progress',
                                       suffix='Complete', length=50)

            np.random.RandomState(seed=epoch).shuffle(self.xtrain_batches)
            np.random.RandomState(seed=epoch).shuffle(self.ytrain_batches)

            for b in range(batches_per_epoch):
                batch_accuracy, op_types, connections = self.train_macro_batch(self.xtrain_batches[b],
                                                                               self.ytrain_batches[b])
                if show_progress:
                    epoch_accuracy = self.macro_graph.retrieve_accuracy(self.macro_graph.macro_epoch_acc)
                    gutil.printProgressBar(b % batches_per_epoch + 1, batches_per_epoch,
                                           prefix='Macro round epoch progress',
                                           suffix=f'Complete. Train accuracy = {epoch_accuracy}', length=50)

    def generate_state_action_reward(self, data_batch):
        x_batch, y_batch = self.xvalid_batches[data_batch], self.yvalid_batches[data_batch]
        self.macro_graph.reset_accuracy(self.macro_graph.ctrl_batch_acc)
        feed_dict = self.controller.create_feed_dict(training=False)
        fetches = [self.controller.op_seq, self.controller.arc_seq]
        op_seq_out, arc_seq_out = self.ctrl_sess.run(fetches, feed_dict)
        if not self.macro_graph.validate_op_types(op_seq_out):
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_batch_acc,
                                             np.ones((128, 10), dtype=np.float32), -np.ones((128,), dtype=np.int32))
            self.macro_graph.update_accuracy(self.macro_graph.ctrl_epoch_acc,
                                             np.ones((128, 10), dtype=np.float32), -np.ones((128,), dtype=np.int32))
        else:
            feed_dict = self.macro_graph.create_feed_dict(x_batch, y_batch, op_seq_out, arc_seq_out)
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

            data_batch += 1

        return op_seqs, arc_seqs, accuracys, data_batch

    def reinforce_from_SARs(self, op_seqs, arc_seqs, accuracies):
        trainable_vars = self.ctrl_sess.graph.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)
        ctrl_batch_grads = [np.zeros(trainable_vars[i].shape) for i in range(len(trainable_vars))]
        for episode in range(self.eps_per_ctrl_batch):
            feed_dict = self.controller.create_feed_dict(training=True,
                                                         prev_accuracy=accuracies[episode],
                                                         moving_average=self.moving_avg,
                                                         prev_op_seq=op_seqs[episode],
                                                         prev_arc_seqs=arc_seqs[episode])
            gradients, loss = self.ctrl_sess.run([self.controller.compute_gradients, self.controller.loss_fn],
                                                 feed_dict)
            ctrl_batch_grads = [np.add(g1, g2) for g1, g2 in zip(ctrl_batch_grads, gradients)]

        feed_dict = self.controller.create_feed_dict(training=False, gradients=ctrl_batch_grads)
        self.ctrl_sess.run(self.controller.apply_gradients, feed_dict)

    def train_controller(self, iter, ctrl_batches=20, report_batch=True, show_progress=True):
        self.macro_graph.reset_accuracy(self.macro_graph.ctrl_epoch_acc)

        data_batch = 0
        if show_progress:
            gutil.printProgressBar(0, ctrl_batches, prefix='Controller round progress ', suffix='Complete', length=50)

        for ctrl_batch in range(ctrl_batches):
            op_seqs, arc_seqs, accuracies, data_batch = self.record_ctrl_batch(data_batch)
            self.reinforce_from_SARs(op_seqs, arc_seqs, accuracies)
            if show_progress:
                acc = self.macro_graph.retrieve_accuracy(self.macro_graph.ctrl_epoch_acc)
                self.macro_graph.reset_accuracy(self.macro_graph.ctrl_epoch_acc)
                gutil.printProgressBar(ctrl_batch % ctrl_batches + 1, ctrl_batches,
                                       prefix='Controller round progress ', suffix=f'Complete. Valid accuracy = {acc}',
                                       length=50)


    def execute_gnas(self, iterations=10000):
        self.macro_sess.run(self.macro_graph.initialise)
        self.ctrl_sess.run(self.controller.initialise)

        for i in range(iterations):
            print(f'Round {i}.')
            self.train_macro_graph(iter=i, epochs=1, report_epoch=True, batches_per_epoch=None)
            self.train_controller(iter=i, ctrl_batches=30, report_batch=True)

        self.macro_sess.close()
        self.ctrl_sess.close()


if __name__ == '__main__':
    # output_depths = [16, 16, 32, 32, 64, 128, 256, 512]
    # output_depths = [16, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512]
    # output_depths = [8, 16, 32, 64, 128, 256]
    # output_depths = [16, 16, 32, 32, 64, 64]
    output_depths = [36] * 6
    batch_size = 32
    c.v1.disable_eager_execution()
    gnas = GNAS(dataset='cifar10', output_depths=output_depths, batch_size=batch_size)
    gnas.execute_gnas(1000)
