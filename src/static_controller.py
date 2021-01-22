from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from tensorflow import compat as c


class Controller(Model):

    def __init__(self, units, num_layers, num_op_types, initializer, batch_size=1, skip_target=0.4, tanh_constant=1.5,
                 temp=5.0):
        super(Controller, self).__init__()
        self.units = units
        self.num_layers = num_layers
        self.num_op_types = num_op_types
        self.batch_size = batch_size
        self.skip_target = skip_target
        self.tanh_constant = tanh_constant
        self.temp = temp
        self.initializer = initializer

        self.optimizer = c.v1.train.AdamOptimizer
        self.learning_rate = 0.0025
        self.op_type_entropy_weight = 0.1
        self.connection_entropy_weight = 0.1
        self.skip_penalty_weight = 0.8

        graph, op_seq, arc_seq, initialise, compute_gradients, apply_gradients, op_logits, loss = self.build_controller()
        self.graph = graph
        self.op_seq = op_seq
        self.arc_seq = arc_seq
        self.initialise = initialise
        self.compute_gradients = compute_gradients
        self.apply_gradients = apply_gradients
        self.op_logits = op_logits
        self.loss_fn = loss

        self.sess = c.v1.Session(graph=self.graph)


    def create_feed_dict(self, training, prev_accuracy=None, moving_average=None, prev_op_seq=None,
                         prev_arc_seqs=None, gradients=None):
        feed_dict = {'input/training:0': 1 if training else 0}
        if prev_accuracy is None:
            prev_accuracy = 0
        if moving_average is None:
            moving_average = 0
        if prev_op_seq is None:
            prev_op_seq = [2] * self.num_layers
        if prev_arc_seqs is None:
            prev_arc_seqs = []
            for i in range(self.num_layers):
                prev_arc_seqs.append([0] * (i))

        feed_dict['input/prev_accuracy:0'] = prev_accuracy
        feed_dict['input/moving_average:0'] = moving_average
        feed_dict['input/prev_op_seq:0'] = prev_op_seq
        for i in range(len(prev_arc_seqs)):
            feed_dict[f'input/prev_arc_seq_layer_{i}:0'] = np.array(prev_arc_seqs[i], np.uint8)

        if gradients is not None:
            for i in range(len(gradients)):
                feed_dict[f'input/gradient_{i}:0'] = gradients[i]

        return feed_dict

    def build_controller(self):
        g = c.v1.Graph()
        with g.as_default():
            optimizer = self.optimizer(learning_rate=self.learning_rate)

            w_lstm = [self.get_variable('w_lstm', 'controller', shape=(2 * self.units, 4 * self.units))]
            w_soft = self.get_variable('w_soft', 'controller', shape=(self.units, self.num_op_types))
            w_emb = self.get_variable('w_emb', 'controller', shape=(self.num_op_types, self.units))
            w_attn_1 = self.get_variable('w_attn_1', 'controller', shape=(self.units, self.units))
            w_attn_2 = self.get_variable('w_attn_2', 'controller', shape=(self.units, self.units))
            v_attn = self.get_variable('v_attn', 'controller', shape=(self.units, 1))
            g_emb = self.get_variable('g_emb', 'controller', shape=(1, self.units))

            with c.v1.variable_scope('input', reuse=c.v1.AUTO_REUSE):
                training = c.v1.placeholder(dtype=tf.uint8, shape=(), name='training')
                prev_accuracy = c.v1.placeholder(dtype=tf.float32, shape=(), name='prev_accuracy')
                moving_average = c.v1.placeholder(dtype=tf.float32, shape=(), name='moving_average')
                prev_op_seq = c.v1.placeholder(dtype=tf.uint8, shape=(self.num_layers,), name='prev_op_seq')
                prev_arc_seqs = []
                gradients = []
                for i in range(self.num_layers):
                    prev_arc_seq = c.v1.placeholder(dtype=tf.uint8, shape=(i,), name=f'prev_arc_seq_layer_{i}')
                    prev_arc_seqs.append(prev_arc_seq)
                train_vars = g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)
                for i in range(len((train_vars))):
                    gradient = c.v1.placeholder(dtype=tf.float32, shape=train_vars[i].shape, name=f'gradient_{i}')
                    gradients.append(gradient)

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

            prev_c = [tf.zeros([1, self.units], tf.float32) for _ in range(self.num_layers)]
            prev_h = [tf.zeros([1, self.units], tf.float32) for _ in range(self.num_layers)]
            inputs = g_emb

            skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target], dtype=tf.float32)

            for layer_id in range(self.num_layers):
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
                else:
                    inputs = g_emb

                anchors.append(next_h[-1])
                anchors_w_1.append(tf.matmul(next_h[-1], w_attn_1))

            op_entropys = tf.stack(op_entropys)
            op_entropys = tf.reduce_sum(op_entropys)
            skip_entropys = tf.stack(skip_entropys)
            skip_entropys = tf.reduce_sum(skip_entropys)
            skip_count = tf.stack(skip_count)
            skip_count = tf.reduce_sum(skip_count)
            skip_penaltys = tf.stack(skip_penaltys)
            skip_penaltys = tf.reduce_sum(skip_penaltys)

            reward = prev_accuracy - moving_average
            # reward += self.op_type_entropy_weight * op_entropys
            # reward += self.skip_penalty_weight * skip_entropys

            sum_probs = tf.reduce_sum([tf.reduce_sum(lp) for lp in log_probs])
            loss = tf.multiply(sum_probs, -reward)
            # loss += self.skip_penalty_weight * skip_penaltys

            compute_gradients = c.v1.gradients(loss, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES))
            apply_gradients = optimizer.apply_gradients(
                zip(gradients, g.get_collection(c.v1.GraphKeys.TRAINABLE_VARIABLES)))
            initialise = c.v1.global_variables_initializer()

        return g, op_seq, arc_seq, initialise, compute_gradients, apply_gradients, op_logits, loss


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


