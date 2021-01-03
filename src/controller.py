from tensorflow.keras import Model
import tensorflow as tf
import numpy as np


class Controller(Model):

    def __init__(self, units, num_layers, num_op_types, initializer, batch_size=1, skip_target=0.4, tanh_constant=2.5,
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

        self.w_lstm = [tf.Variable(self.initializer(shape=(2 * self.units, 4 * self.units)))]
        self.w_soft = tf.Variable(self.initializer(shape=(self.units, num_op_types)))
        self.w_emb = tf.Variable(self.initializer(shape=(num_op_types, units)))
        self.w_attn_1 = tf.Variable(self.initializer(shape=(units, units)))
        self.w_attn_2 = tf.Variable(self.initializer(shape=(units, units)))
        self.v_attn = tf.Variable(self.initializer(shape=(units, 1)))
        self.g_emb = tf.Variable(self.initializer(shape=(1, units)))

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

    def call(self, input_arg=None, **kwargs):
        anchors = []
        anchors_w_1 = []

        op_seq = []
        arc_seq = []
        op_entropys = []
        skip_entropys = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        prev_c = [tf.zeros([1, self.units], tf.float32) for _ in range(self.num_layers)]
        prev_h = [tf.zeros([1, self.units], tf.float32) for _ in range(self.num_layers)]
        inputs = self.g_emb

        skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target], dtype=tf.float32)

        for layer_id in range(self.num_layers):
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            logit = tf.matmul(next_h[-1], self.w_soft)

            if self.temp is not None:
                logit /= self.temp
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)

            op_type_id = tf.random.categorical(logits=logit, num_samples=1, dtype=tf.int32)
            op_type_id = tf.reshape(op_type_id, [1])
            op_seq.append(op_type_id)
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=op_type_id)
            log_probs.append(log_prob)

            entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
            op_entropys.append(entropy)

            inputs = tf.nn.embedding_lookup(self.w_emb, op_type_id)

            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h,
                                             self.w_lstm)  # TODO: this line is maybe causing issues? unlikely...
            prev_c, prev_h = next_c, next_h

            if layer_id > 0:
                query = tf.concat(anchors_w_1, axis=0)
                query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
                query = tf.matmul(query, self.v_attn)
                logit = tf.concat([-query, query], axis=1)

                if self.temp is not None:
                    logit /= self.temp
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)

                skip = tf.random.categorical(logits=logit, num_samples=1)
                skip = tf.cast(skip, tf.int32)
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
                inputs = self.g_emb

            anchors.append(next_h[-1])
            anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

        op_entropys = tf.stack(op_entropys)
        op_entropys = tf.reduce_sum(op_entropys)
        skip_entropys = tf.stack(skip_entropys)
        skip_entropys = tf.reduce_sum(skip_entropys)
        skip_count = tf.stack(skip_count)
        skip_count = tf.reduce_sum(skip_count)
        skip_penaltys = tf.stack(skip_penaltys)
        skip_penaltys = tf.reduce_sum(skip_penaltys)

        return op_seq, arc_seq, log_probs, op_entropys, skip_entropys, skip_count, skip_penaltys

    def generate_random_architecture(self, num_op_types=None, num_layers=6):
        if num_op_types is None:
            num_op_types = self.num_op_types
        op_actions = np.random.choice(len(num_op_types), num_layers)
        op_actions = np.concatenate([op_actions, np.random.choice(4, 1)])
        connections = [np.random.choice(2, i) for i in range(1, num_layers + 1)]
        for connection in connections:
            connection[-1] = 1
        return op_actions, connections


