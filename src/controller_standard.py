# from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from tensorflow import compat as c
# import dgl
# import dgl.nn.tensorflow as dgltf
from functools import partial
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Controller(object):

    def __init__(self, n_op_layers, num_op_types, initializer, input_shape, batch_size=1, **kwargs):
        super(Controller, self).__init__()
        self.units = kwargs.get('ctrl_units')
        self.n_lstm_layers = kwargs.get('ctrl_lstm_layers')
        self.n_op_layers = n_op_layers
        self.num_op_types = num_op_types
        self.batch_size = batch_size
        self.skip_target = kwargs.get('ctrl_skip_target')
        self.tanh_constant = kwargs.get('ctrl_tanh_constant')
        self.temp = kwargs.get('ctrl_temp')
        self.initializer = initializer

        #TEMP
        self.gnn_layer_dims = kwargs.get('ctrl_gnn_layer_dims')
        self.input_shape = input_shape
        self.output_depths = kwargs.get('output_depths')
        self.pool_layers = kwargs.get('pool_layers')

        self.optimizer = kwargs.get('ctrl_optimizer')
        # self.learning_rate = 0.0025
        self.op_type_entropy_weight = kwargs.get('ctrl_op_type_entropy_weight')
        self.connection_entropy_weight = kwargs.get('ctrl_connection_entropy_weight')
        self.skip_penalty_weight = kwargs.get('ctrl_skip_penalty_weight')

        self.graph, self.op_seq, self.arc_seq, self.initialise, self.compute_gradients, \
        self.apply_gradients, self.op_logits, self.loss_fn, self.reward_fn, self.list_of_things = \
            self.build_controller()

        self.sess = c.v1.Session(graph=self.graph)


    def create_feed_dict(self, training, prev_accuracy=None, moving_average=None, prev_op_seq=None,
                         prev_arc_seqs=None, gradients=None, learning_rate=None):
        feed_dict = {'input/training:0': 1 if training else 0}
        if prev_accuracy is None:
            prev_accuracy = 0
        if moving_average is None:
            moving_average = 0
        if prev_op_seq is None:
            prev_op_seq = [2] * self.n_op_layers
        if prev_arc_seqs is None:
            prev_arc_seqs = []
            for i in range(self.n_op_layers):
                prev_arc_seqs.append([0] * (i))
        if learning_rate is None:
            learning_rate = 0

        feed_dict['input/prev_accuracy:0'] = prev_accuracy
        feed_dict['input/moving_average:0'] = moving_average
        feed_dict['input/prev_op_seq:0'] = prev_op_seq
        feed_dict['input/output_depths:0'] = np.array(self.output_depths)[:, np.newaxis]
        feed_dict['input/input_shape:0'] = np.array(self.input_shape)[:, np.newaxis]
        for i in range(len(prev_arc_seqs)):
            feed_dict[f'input/prev_arc_seq_layer_{i}:0'] = np.array(prev_arc_seqs[i], np.uint8)

        if gradients is not None:
            for i in range(len(gradients)):
                feed_dict[f'input/gradient_{i}:0'] = gradients[i]

        feed_dict['input/learning_rate:0'] = learning_rate

        return feed_dict

    def build_controller(self):
        g = c.v1.Graph()
        with g.as_default():

            w_lstm = []
            for i in range(self.n_lstm_layers):
                w_lstm.append(self.get_variable(f'w_lstm_{i}', 'controller',
                                                shape=(2 * self.units, 4 * self.units)))

            w_soft = self.get_variable('w_soft', 'controller', shape=(self.units, self.num_op_types))
            w_emb = self.get_variable('w_emb', 'controller', shape=(self.num_op_types, self.units))
            w_attn_1 = self.get_variable('w_attn_1', 'controller', shape=(self.units, self.units))
            w_attn_2 = self.get_variable('w_attn_2', 'controller', shape=(self.units, self.units))
            v_attn = self.get_variable('v_attn', 'controller', shape=(self.units, 1))
            g_emb = self.get_variable('g_emb', 'controller', shape=(1, self.units))

            # with c.v1.variable_scope('gnn', reuse=c.v1.AUTO_REUSE):
            #     gnn_adj = c.v1.get_variable('gnn_adj',
            #                                 initializer=lambda: tf.constant_initializer(0)(
            #                                     shape=(self.n_op_layers, self.n_op_layers), dtype=tf.uint8),
            #                                 trainable=False)
            #     gnn_layer_0 = c.v1.get_variable('gnn_layer_0',
            #                                     initializer=lambda: tf.constant_initializer(0.0)(
            #                                         shape=(self.n_op_layers, self.gnn_layer_dims[0]), dtype=tf.float32),
            #                                     trainable=False)
            #     gnn_w1 = c.v1.get_variable('gnn_w1', initializer=self.initializer,
            #                                shape=(self.gnn_layer_dims[0], self.gnn_layer_dims[1]))
            #     gnn_w2 = c.v1.get_variable('gnn_w2', initializer=self.initializer,
            #                                shape=(self.gnn_layer_dims[1], self.gnn_layer_dims[2]))
            #     gnn_b1 = c.v1.get_variable('gnn_b1',
            #                                initializer=lambda: tf.constant_initializer(0.0)(
            #                                    shape=(1, self.gnn_layer_dims[1]), dtype=tf.float32))
            #     gnn_b2 = c.v1.get_variable('gnn_b2',
            #                                initializer=lambda: tf.constant_initializer(0.0)(
            #                                    shape=(1, self.gnn_layer_dims[2]), dtype=tf.float32))
            #     gnn_attn = c.v1.get_variable('gnn_attn', initializer=self.initializer,
            #                                  shape=(self.units, self.units - self.gnn_layer_dims[-1]))


            with c.v1.variable_scope('input', reuse=c.v1.AUTO_REUSE):
                output_depths = c.v1.placeholder(dtype=tf.int32, shape=(self.n_op_layers, 1), name='output_depths')
                input_shape = c.v1.placeholder(dtype=tf.int32, shape=(4, 1), name='input_shape')
                training = c.v1.placeholder(dtype=tf.uint8, shape=(), name='training')
                prev_accuracy = c.v1.placeholder(dtype=tf.float32, shape=(), name='prev_accuracy')
                moving_average = c.v1.placeholder(dtype=tf.float32, shape=(), name='moving_average')
                prev_op_seq = c.v1.placeholder(dtype=tf.uint8, shape=(self.n_op_layers,), name='prev_op_seq')
                prev_arc_seqs = []
                gradients = []
                for i in range(self.n_op_layers):
                    prev_arc_seq = c.v1.placeholder(dtype=tf.uint8, shape=(i,), name=f'prev_arc_seq_layer_{i}')
                    prev_arc_seqs.append(prev_arc_seq)
                train_vars = g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)
                for i in range(len((train_vars))):
                    gradient = c.v1.placeholder(dtype=tf.float32, shape=train_vars[i].shape, name=f'gradient_{i}')
                    gradients.append(gradient)
                learning_rate = c.v1.placeholder(dtype=tf.float32, shape=(), name='learning_rate')


            anchors = []
            anchors_w_1 = []

            op_seq = []
            arc_seq = [[]]
            op_logits = []
            op_entropys = []
            skip_entropys = []
            log_probs = []
            skip_count = []
            skip_penaltys = []

            prev_c = [tf.zeros([1, self.units], tf.float32) for _ in range(self.n_op_layers)]
            prev_h = [tf.zeros([1, self.units], tf.float32) for _ in range(self.n_op_layers)]
            inputs = g_emb

            skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target], dtype=tf.float32)

            # stem op always sets channels to match first output depth
            # channels = output_depths[0]
            # -2 always height or width regardless of whether channels first, last, or not present (assumes square)
            # spatial_dim = input_shape[-2]

            list_of_things = []

            for layer_id in range(self.n_op_layers):
                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, w_lstm)
                logit = tf.matmul(next_h[-1], w_soft)

                op_logits.append(logit)

                if self.temp is not None:
                    logit /= self.temp
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)

                op_type_id = tf.cond(tf.equal(training, tf.constant(1, tf.uint8)),
                                     lambda: tf.cast(prev_op_seq[layer_id], tf.int32),
                                     lambda: tf.cast(tf.random.categorical(logits=logit, num_samples=1), tf.int32))

                op_type_id = tf.reshape(op_type_id, [1])
                op_seq.append(tf.reshape(op_type_id, ()))

                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=op_type_id)
                log_probs.append(log_prob)
                entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
                op_entropys.append(entropy)

                # inputs = tf.nn.embedding_lookup(w_emb, op_type_id)
                # tf.nn.embedding_lookup produces gradient as an IndexedSliceValue, no convenient way to batch and apply
                # so manually do the embedding lookup by one hot encoding op type and
                # taking dot product with embedding table
                onehot_op = tf.one_hot(indices=op_type_id, depth=self.num_op_types)
                inputs = tf.tensordot(onehot_op, w_emb, axes=[[1], [0]])



                # get the layer's output dimension from feed dict
                # and embed it using... tensordot?
                # in_ch = tf.cast(tf.reshape(channels, shape=(1, 1)), dtype=tf.float32)
                # out_ch = tf.cast(tf.reshape(output_depths[layer_id], shape=(1, 1)), dtype=tf.float32)
                # channels = out_ch
                # in_spatial = tf.cast(tf.reshape(spatial_dim, shape=(1, 1)), dtype=tf.float32)
                # out_spatial = spatial_dim // 2 if layer_id in self.pool_layers else spatial_dim
                # out_spatial = tf.cast(tf.reshape(out_spatial, shape=(1, 1)), dtype=tf.float32)
                # spatial_dim = out_spatial
                #
                # # generate node feature vector
                # scatter_indices = [[layer_id, i] for i in range(self.gnn_layer_dims[0])]
                # scatter_updates = tf.concat([in_ch, out_ch, in_spatial, out_spatial, onehot_op], axis=1)
                # gnn_layer_0 = tf.tensor_scatter_nd_update(gnn_layer_0,
                #                                           scatter_indices,
                #                                           tf.reshape(scatter_updates, shape=(self.gnn_layer_dims[0],)))

                # gnn_layer_0[layer_id] = tf.concat([in_ch, out_ch, in_spatial, out_spatial, tf.transpose(onehot_op)], axis=0)
                # add the identity matrix element for self loop (no other connections yet)
                # scatter_indices = [[layer_id, layer_id]]
                # scatter_updates = [1]
                # gnn_adj = tf.tensor_scatter_nd_update(gnn_adj, scatter_indices, scatter_updates)
                # # gnn_adj[layer_id][layer_id] = 1
                # gnn_layer_1 = tf.matmul(tf.cast(gnn_adj, tf.float32), gnn_layer_0)
                # gnn_layer_1 = tf.matmul(gnn_layer_1, gnn_w1)
                # gnn_layer_1 = tf.add(gnn_layer_1, gnn_b1)
                # gnn_layer_1 = tf.nn.relu(gnn_layer_1)
                #
                # gnn_layer_2 = tf.matmul(tf.cast(gnn_adj, tf.float32), gnn_layer_1)
                # gnn_layer_2 = tf.matmul(gnn_layer_2, gnn_w2)
                # gnn_layer_2 = tf.add(gnn_layer_2, gnn_b2)
                # gnn_layer_2 = tf.nn.relu(gnn_layer_2)
                #
                # gnn_out = tf.expand_dims(tf.reduce_mean(gnn_layer_2, axis=0), axis=0)
                # current_node = tf.expand_dims(gnn_layer_2[layer_id], axis=0)


                # standardise gnn output to lstm mean and std
                # inputs_mean = tf.reduce_mean(inputs)
                # inputs_std = tf.math.reduce_std(inputs)
                # inputs_std = tf.add(inputs_std, 1e-5)
                # inputs = tf.subtract(inputs, inputs_mean)
                # inputs = tf.divide(inputs, inputs_std)
                #
                # gnn_out_mean = tf.reduce_mean(gnn_out)
                # gnn_out_std = tf.math.reduce_std(gnn_out)
                # gnn_out_std = tf.math.add(gnn_out_std, 1e-5)
                # gnn_out = tf.subtract(gnn_out, gnn_out_mean)
                # gnn_out = tf.divide(gnn_out, gnn_out_std)
                #
                # current_node_mean = tf.reduce_mean(current_node)
                # current_node_std = tf.math.reduce_std(current_node)
                # current_node_std = tf.add(current_node_std, 1e-5)
                # current_node = tf.subtract(current_node, current_node_mean)
                # current_node = tf.divide(current_node, current_node_std)

                # gnn_out = tf.subtract(gnn_out, gnn_out_mean)
                # if layer_id == 1:
                #     list_of_things.extend([inputs_mean, inputs_std, gnn_out_mean, gnn_out_std, gnn_out])
                # gnn_out = tf.multiply(gnn_out, tf.divide(inputs_std, gnn_out_std))
                # if layer_id == 1:
                #     list_of_things.append(gnn_out)
                # gnn_out = tf.add(gnn_out, inputs_mean)
                # if layer_id == 1:
                #     list_of_things.append(gnn_out)
                # current_node = tf.subtract(current_node, current_node_mean)
                # current_node = tf.multiply(current_node, tf.divide(inputs_std, current_node_std))
                # current_node = tf.add(current_node, inputs_mean)




                # inputs = tf.concat([inputs, current_node, gnn_out], axis=1)


                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, w_lstm)
                prev_c, prev_h = next_c, next_h

                if layer_id > 0:
                    query = tf.concat(anchors_w_1, axis=0)
                    query = tf.tanh(query + tf.matmul(next_h[-1], w_attn_2))
                    query = tf.matmul(query, v_attn)
                    logit = tf.concat([-query, query], axis=1)

                    if self.temp is not None:
                        logit /= self.temp
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)

                    skip = tf.cond(tf.equal(training, tf.constant(1, tf.uint8)),
                                   lambda: tf.cast(prev_arc_seqs[layer_id], tf.int32),
                                   lambda: tf.cast(tf.random.categorical(logits=logit, num_samples=1), tf.int32))
                    skip = tf.reshape(skip, [layer_id])
                    arc_seq.append(skip)

                    # paddings = tf.expand_dims(tf.constant([0, self.n_op_layers - layer_id]), axis=0)
                    # adj_row = tf.pad(skip, paddings, mode='CONSTANT', constant_values=0)
                    # adj_col = tf.transpose(adj_row)
                    # gnn_adj[layer_id] = adj_row
                    # gnn_adj[:, layer_id] = adj_col

                    # row_indices = [[layer_id, i] for i in range(layer_id)]
                    # col_indices = [[i, layer_id] for i in range(layer_id)]
                    # gnn_adj = tf.tensor_scatter_nd_update(gnn_adj,
                    #                                       row_indices,
                    #                                       tf.cast(skip, tf.uint8))
                    # gnn_adj = tf.tensor_scatter_nd_update(gnn_adj,
                    #                                       col_indices,
                    #                                       tf.cast(skip, tf.uint8))




                    skip_prob = tf.sigmoid(logit)
                    kl = skip_prob * tf.math.log(skip_prob / skip_targets)
                    kl = tf.reduce_sum(kl)
                    skip_penaltys.append(kl)

                    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=skip)
                    log_probs.append(tf.reduce_sum(log_prob, keepdims=True))

                    entropy = tf.stop_gradient(tf.reduce_sum(log_prob * tf.exp(-log_prob), keepdims=True))
                    skip_entropys.append(entropy)

                    skip = tf.cast(skip, tf.float32)
                    skip = tf.reshape(skip, [1, layer_id])
                    skip_count.append(tf.reduce_sum(skip))
                    inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
                    inputs /= (1.0 + tf.reduce_sum(skip))

                    # inputs = tf.matmul(inputs, gnn_attn)

                    # gnn_layer_1 = tf.matmul(tf.cast(gnn_adj, tf.float32), gnn_layer_0)
                    # gnn_layer_1 = tf.matmul(gnn_layer_1, gnn_w1)
                    # gnn_layer_1 = tf.add(gnn_layer_1, gnn_b1)
                    # gnn_layer_1 = tf.nn.relu(gnn_layer_1)
                    #
                    # gnn_layer_2 = tf.matmul(tf.cast(gnn_adj, tf.float32), gnn_layer_1)
                    # gnn_layer_2 = tf.matmul(gnn_layer_2, gnn_w2)
                    # gnn_layer_2 = tf.add(gnn_layer_2, gnn_b2)
                    # gnn_layer_2 = tf.nn.relu(gnn_layer_2)
                    #
                    # gnn_out = tf.expand_dims(tf.reduce_mean(gnn_layer_2, axis=0), axis=0)
                    #
                    # inputs_mean = tf.reduce_mean(inputs)
                    # inputs_std = tf.math.reduce_std(inputs)
                    # inputs_std = tf.add(inputs_std, 1e-5)
                    # inputs = tf.subtract(inputs, inputs_mean)
                    # inputs = tf.divide(inputs, inputs_std)
                    #
                    # gnn_out_mean = tf.reduce_mean(gnn_out)
                    # gnn_out_std = tf.math.reduce_std(gnn_out)
                    # gnn_out_std = tf.math.add(gnn_out_std, 1e-5)
                    # gnn_out = tf.subtract(gnn_out, gnn_out_mean)
                    # gnn_out = tf.divide(gnn_out, gnn_out_std)
                    #
                    # inputs = tf.concat([inputs, gnn_out], axis=1)

                else:
                    inputs = g_emb
                    # inputs = tf.matmul(inputs, gnn_attn)
                    # inputs = tf.concat([inputs, current_node], axis=1)

                anchors.append(next_h[-1])
                anchors_w_1.append(tf.matmul(next_h[-1], w_attn_1))

            op_entropys = tf.stack(op_entropys)
            op_entropys = tf.reduce_sum(op_entropys)
            skip_entropys = tf.stack(skip_entropys)
            skip_entropys = tf.reduce_sum(skip_entropys)
            skip_count = tf.stack(skip_count)
            skip_count = tf.reduce_sum(skip_count)
            skip_penaltys = tf.stack(skip_penaltys)
            skip_penaltys = tf.reduce_mean(skip_penaltys)

            reward = prev_accuracy
            # reward += tf.cast(tf.reduce_sum(op_seq), tf.float32)
            reward += self.op_type_entropy_weight * op_entropys
            reward += self.connection_entropy_weight * skip_entropys
            # reward -= moving_average

            sum_probs = tf.reduce_sum([tf.reduce_sum(lp) for lp in log_probs])
            loss = tf.multiply(sum_probs, -(reward - moving_average))
            loss += self.skip_penalty_weight * skip_penaltys

            compute_gradients = c.v1.gradients(loss, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))

            optimizer = self.optimizer(learning_rate=learning_rate)
            compute_gradients2 = optimizer.compute_gradients(loss, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))
            apply_gradients = optimizer.apply_gradients(
                zip(gradients, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)))
            apply_gradients = optimizer.minimize(-loss)
            initialise = c.v1.global_variables_initializer()

            list_of_things.append({
                'op_entropys': op_entropys,
                'skip_entroys': skip_entropys,
                'skip_count': skip_count,
                'skip_penaltys': skip_penaltys,
                'reward': reward,
                'baseline': moving_average,
                'sum_probs': sum_probs,
                'loss': loss
            })

        return g, op_seq, arc_seq, initialise, compute_gradients, apply_gradients, op_logits, loss, reward, list_of_things


    def get_variable(self, var_name, scope_name, shape, dtype=tf.float32, initialiser=None,
                     reuse=c.v1.AUTO_REUSE, regulariser=None, trainable=True):
        if initialiser is None:
            initialiser = self.initializer
        # if regulariser is None:
        #     regulariser = lambda w: tf.reduce_mean(w**2)

        with c.v1.variable_scope(scope_name, reuse=reuse):
            var = c.v1.get_variable(var_name, initializer=initialiser(shape, dtype),
                                    regularizer=regulariser, trainable=trainable)
        return var

    def lstm(self, x, prev_c, prev_h, w):
        ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
        i, f, o, g = tf.split(ifog, 4, axis=1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        next_c = i * g + f * prev_c
        next_h = o * tf.tanh(next_c)
        return next_c, next_h

    def stack_lstm(self, x, prev_c, prev_h, w):
        next_c, next_h = [], []
        for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_c, curr_h = self.lstm(inputs, _c, _h, _w)
            next_c.append(curr_c)
            next_h.append(curr_h)
        return next_c, next_h

    def generate_random_architecture(self, num_op_types=None, num_layers=6):
        if num_op_types is None:
            num_op_types = self.num_op_types
        op_actions = np.random.choice(num_op_types, num_layers)
        # op_actions = np.concatenate([op_actions, np.random.choice(4, 1)])
        connections = [np.random.choice(2, i) for i in range(1, num_layers)]
        for connection in connections:
            connection[-1] = 1
        return op_actions[:, np.newaxis], connections


