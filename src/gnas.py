from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow import GradientTape
import tensorflow as tf
import numpy as np
import controller as ctrl
import macro_graph as mg
import gnas_data as gdata
import gnas_utils as gutil

class GNAS(object):
    def __init__(self, dataset, output_depths, batch_size=128, controller_units=100, controller_lr=0.0035,
                 eps_per_ctrl_batch=30):
        super(GNAS, self).__init__()
        if output_depths is None:
            output_depths = [16, 16, 32, 32, 64, 128, 256, 512]
        self.output_depths = output_depths
        self.controller_units = controller_units
        self.initializer = RandomUniform(-0.1, 0.1)

        self.xtrain_batches, self.ytrain_batches, self.xvalid_batches, self.yvalid_batches = \
            gdata.prepare_data(dataset, batch_size=batch_size, valid_size=0.2)

        self.controller = ctrl.Controller(units=controller_units,
                                          num_layers=len(output_depths),
                                          num_op_types=len(mg.OP_TYPES),
                                          initializer=self.initializer)
        self.macro_graph = mg.MacroGraph(input_shape=self.xtrain_batches[0].shape, output_depths=output_depths,
                                         regularization_rate=5e-4)

        self.ctrl_optimizer = Adam(learning_rate=controller_lr)
        self.eps_per_ctrl_batch = eps_per_ctrl_batch

        self.macro_loss_fn = SparseCategoricalCrossentropy()
        self.accuracy_fn = SparseCategoricalAccuracy()
        self.epoch_accuracy_fn = SparseCategoricalAccuracy()

        self.macro_lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9,
                                                  staircase=True)
        self.macro_optimizer = Adam(learning_rate=self.macro_lr_schedule)

        # self.moving_avg = 1 / max(self.ytrain_batches)   # TODO: check
        self.moving_avg = 0.1   # TODO: check
        self.exp_ma_alpha = 2 / (len(self.xtrain_batches) + 1)

        self.op_type_entropy_weight = 0.1
        self.connection_entropy_weight = 0.1
        self.skip_penalty_weight = 0.8

    def sample_valid_architecture(self):
        while True:
            op_types, connect_seq, log_props, op_entropys, skip_entropys, skip_count, skip_penaltys = \
                self.controller('placeholder')
            op_types = np.array(op_types)[:, 0]
            connections = [[1]]
            connections.extend([list(np.array(con)) for con in connect_seq])
            for c in range(1, len(connections)):
                connections[c].append(1)
            if self.macro_graph.validate_op_types(op_types):
                return op_types, connections, log_props, op_entropys, skip_entropys, skip_count, skip_penaltys

    def train_macro_batch(self, x_batch, y_batch):
        op_types, connections, log_props, op_entropys, skip_entropys, skip_count, skip_penaltys = \
            self.sample_valid_architecture()

        # op_types = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # connections = [[1 for i in range(j)] for j in range(1, 13)]


        model = self.macro_graph.build_model(op_types, connections)
        with GradientTape() as tape:
            model_out = model(x_batch)
            loss = self.macro_loss_fn(y_batch, model_out)
            gradients = tape.gradient(loss, model.trainable_variables)
        self.macro_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.epoch_accuracy_fn.update_state(y_batch, model_out)
        self.accuracy_fn.update_state(y_batch, model_out)
        accuracy = self.accuracy_fn.result()
        self.accuracy_fn.reset_states()
        self.moving_avg = (self.exp_ma_alpha * accuracy) + (1 - self.exp_ma_alpha) * self.moving_avg

        return accuracy, loss, op_types, connections


    def train_macro_graph(self, iter, epochs=1, report_epoch=True, show_progress=True):
        for epoch in range(epochs):
            np.random.RandomState(seed=epoch).shuffle(self.xtrain_batches)
            np.random.RandomState(seed=epoch).shuffle(self.ytrain_batches)

            # for b in range(2):
            for b in range(len(self.xtrain_batches)):
                accuracy, loss, op_types, connections = self.train_macro_batch(self.xtrain_batches[b],
                                                                               self.ytrain_batches[b])
                if show_progress:
                    gutil.printProgressBar(b+1, len(self.xtrain_batches), prefix='Macro graph epoch progress',
                                           suffix=f'Epoch accuracy={self.epoch_accuracy_fn.result()}', length=50)
            if report_epoch:
                print(f'Trained macro graph, iter={iter}, epoch={epoch}\n'
                      f'epoch accuracy={self.epoch_accuracy_fn.result()}\nmoving_avg={self.moving_avg}\n'
                      f'last sampled arc: accuracy={accuracy}\nop_actions={op_types}\nconnections={connections}\n\n')

    def train_controller_batch(self, xbatch, ybatch):
        with GradientTape() as tape:
            op_types, connect_seq, log_probs, op_entropys, skip_entropys, skip_count, skip_penaltys = self.controller(0)
            op_types = np.array(op_types)[:, 0]
            connections = [[1]]
            connections.extend([list(np.array(con)) for con in connect_seq])
            for i in range(1, len(connections)):
                connections[i].append(1)
            if not self.macro_graph.validate_op_types(op_types):
                self.accuracy_fn.update_state(np.ones((128, 1)), np.zeros((128, 1)))
                self.epoch_accuracy_fn.update_state(np.ones((128, 1)), np.zeros((128, 1)))
            else:
                model = self.macro_graph.build_model(op_types, connections)
                model_out = model(xbatch)
                self.accuracy_fn.update_state(ybatch, model_out)
                self.epoch_accuracy_fn.update_state(ybatch, model_out)

            accuracy = self.accuracy_fn.result()
            self.accuracy_fn.reset_states()
            self.moving_avg = (self.exp_ma_alpha * accuracy) + (1 - self.exp_ma_alpha) * self.moving_avg

            reward = accuracy - self.moving_avg
            reward += self.op_type_entropy_weight * op_entropys
            reward += self.connection_entropy_weight * skip_entropys

            sum_probs = tf.reduce_sum([tf.reduce_sum(lp) for lp in log_probs])
            loss = tf.multiply(sum_probs, reward)
            loss += self.skip_penalty_weight * skip_penaltys
        return tape.gradient(loss, self.controller.trainable_variables), accuracy, op_types, connections

    def train_controller(self, iter, steps=300, report_batch=True, show_progress=True):
        for step in range(steps):
            if step % len(self.xvalid_batches) == 0:
                self.xvalid_batches, self.yvalid_batches = gdata.shuffle_xy(self.xvalid_batches, self.yvalid_batches)
                batch_index = 0

            if step % self.eps_per_ctrl_batch == 0:
                batch_grads = [tf.zeros_like(self.controller.trainable_variables[i]) for i in
                               range(len(self.controller.trainable_variables))]
                if show_progress:
                    gutil.printProgressBar(0, self.eps_per_ctrl_batch, prefix='Controller batch progress',
                                           suffix='Complete', length=50)

            grads, accuracy, op_types, connections = self.train_controller_batch(
                self.xtrain_batches[batch_index], self.ytrain_batches[batch_index])
            batch_grads = [tf.math.add(g1, g2) for g1, g2 in zip(batch_grads, grads)]
            if show_progress:
                gutil.printProgressBar(step % self.eps_per_ctrl_batch + 1, self.eps_per_ctrl_batch,
                                       prefix='Controller batch progress', suffix='Complete', length=50)

            if (step + 1) % self.eps_per_ctrl_batch == 0:
                self.ctrl_optimizer.apply_gradients(zip(batch_grads, self.controller.trainable_variables))
                if report_batch:
                    print(f'\n\nTraining controller, iter={iter}, next_step={step+1}\n'
                          f'epoch accuracy={self.epoch_accuracy_fn.result()}\nmoving_avg={self.moving_avg}\n'
                          f'last sampled arc: accuracy={accuracy}\nop_actions={op_types}\nconnections={connections}')



    def execute_gnas(self, iterations=10000):
        for i in range(iterations):
            self.epoch_accuracy_fn.reset_states()
            self.train_macro_graph(iter=i, epochs=2, report_epoch=True)
            self.train_controller(iter=i, steps=300, report_batch=True)


if __name__ == '__main__':
    # output_depths = [16, 16, 32, 32, 64, 128, 256, 512]
    output_depths = [16, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512]
    gnas = GNAS(dataset='cifar10', output_depths=output_depths)
    gnas.execute_gnas(1000)
