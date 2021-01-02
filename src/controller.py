import tensorflow as tf
import numpy as np
import macro_graph as mg

def create_controller(units, batch_size, n_layers):

    # class BahdanauAttention(tf.keras.layers.Layer):
    #     def __init__(self, units):
    #         super(BahdanauAttention, self).__init__()
    #         self.W1 = tf.keras.layers.Dense(units)
    #         self.W2 = tf.keras.layers.Dense(units)
    #         self.V = tf.keras.layers.Dense(1)
    #
    #     def

    class Controller(tf.keras.Model):
        def __init__(self, units, batch_size, n_layers, n_op_types=5, skip_target=0.4,
                     tanh_constant=2.5, temperature=5.0):
            super(Controller, self).__init__()
            self.units = units
            self.batch_size = batch_size
            self.n_layers = n_layers
            self.skip_target = skip_target
            self.tanh_constant = tanh_constant
            self.temperature = temperature
            # self.initializer = tf.keras.initializers.GlorotUniform()
            self.initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
            # self.hidden_state = tf.zeros(shape=(batch_size, units))
            # self.cell_state = tf.zeros(shape=(batch_size, units))
            # self.lstm = tf.keras.layers.LSTM(units=self.units, return_sequences=False, return_state=True)
            self.op_type_mlp = tf.keras.layers.Dense(units=n_op_types, activation='softmax')
            # self.op_type_mlp = tf.keras.layers.Dense(units=n_op_types)
            self.op_type_embedding = tf.keras.layers.Embedding(input_dim=n_op_types, output_dim=units)
            self.bahdanau_W1 = tf.keras.layers.Dense(units)
            self.bahdanau_W2 = tf.keras.layers.Dense(units)
            self.bahdanau_V = tf.keras.layers.Dense(1)
            # self.w_lstm = [self.initializer(shape=(2 * self.units, 4 * self.units))]
            # self.w_soft = self.initializer(shape=(self.units, n_op_types))
            # self.w_emb = self.initializer(shape=(n_op_types, units))
            # self.w_attn_1 = self.initializer(shape=(units, units))
            # self.w_attn_2 = self.initializer(shape=(units, units))
            # self.v_attn = self.initializer(shape=(units, 1))
            # self.g_emb = self.initializer(shape=(1, units))
            # self.grad_variables = [self.w_lstm, self.w_soft, self.w_emb, self.w_attn_1, self.w_attn_2, self.v_attn, self.g_emb]
            self.w_lstm = [tf.Variable(self.initializer(shape=(2 * self.units, 4 * self.units)))]
            self.w_soft = tf.Variable(self.initializer(shape=(self.units, n_op_types)))
            self.w_emb = tf.Variable(self.initializer(shape=(n_op_types, units)))
            self.w_attn_1 = tf.Variable(self.initializer(shape=(units, units)))
            self.w_attn_2 = tf.Variable(self.initializer(shape=(units, units)))
            self.v_attn = tf.Variable(self.initializer(shape=(units, 1)))
            self.g_emb = tf.Variable(self.initializer(shape=(1, units)))
            # self.grad_variables = [self.w_lstm, self.w_soft, self.w_emb, self.w_attn_1, self.w_attn_2, self.v_attn, self.g_emb]

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

        def call(self, input_arg, **kwargs):
            anchors = []
            anchors_w_1 = []

            op_seq = []
            arc_seq = []
            op_entropys = []
            skip_entropys = []
            log_probs = []
            skip_count = []
            skip_penaltys = []

            prev_c = [tf.zeros([1, self.units], tf.float32) for _ in range(self.n_layers)]
            prev_h = [tf.zeros([1, self.units], tf.float32) for _ in range(self.n_layers)]
            inputs = self.g_emb

            skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target], dtype=tf.float32)

            for layer_id in range(self.n_layers):
                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                logit = tf.matmul(next_h[-1], self.w_soft)
                op_type_id = tf.random.categorical(logits=logit, num_samples=1, dtype=tf.int32)
                op_type_id = tf.reshape(op_type_id, [1])
                op_seq.append(op_type_id)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=op_type_id)
                log_probs.append(log_prob)

                entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
                op_entropys.append(entropy)

                inputs = tf.nn.embedding_lookup(self.w_emb, op_type_id)

                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)  # TODO: this line is maybe causing issues? unlikely...
                prev_c, prev_h = next_c, next_h

                if layer_id > 0:
                    query = tf.concat(anchors_w_1, axis=0)
                    query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
                    query = tf.matmul(query, self.v_attn)
                    logit = tf.concat([-query, query], axis=1)
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

        def call0(self, input_arg, **kwargs):
            # prev_state_c = []
            # prev_state_h = []

            op_actions = []
            connections = []
            log_probs = []

            skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target], dtype=tf.float32)
            skip_penalties = []
            skip_count = []

            anchors = []
            anchors_W1 = []

            next_input = tf.expand_dims(input_arg, 1)

            for L in range(self.n_layers):
                output, state_h, state_c = self.lstm(next_input)  # output == state_h
                prev_h, prev_c = state_h, state_c  # TODO: ??
                op_logits = self.op_type_mlp(state_h)
                op_type_id = tf.random.categorical(logits=op_logits, num_samples=1, dtype=tf.int32)
                op_actions.append(tf.reshape(op_type_id, shape=1))
                log_prob = tf.math.log(op_logits)
                log_probs.append(log_prob)

                next_input = self.op_type_embedding(op_type_id)
                output, state_h, state_c = self.lstm(next_input)
                prev_h, prev_c = state_h, state_c  # TODO: ??

                if L > 0:
                    query = tf.concat(anchors_W1, axis=0)
                    score = self.bahdanau_V(tf.nn.tanh(query + self.bahdanau_W2(state_h)))
                    skip_logits = tf.concat([-score, score], axis=1)

                    skip = tf.random.categorical(logits=skip_logits, num_samples=1, dtype=tf.int32)
                    skip = tf.reshape(skip, L)
                    connections.append(skip)

                    skip_prob = tf.sigmoid(skip_logits)
                    KL = skip_prob * tf.math.log(skip_prob / skip_targets)
                    KL = tf.reduce_sum(KL)
                    skip_penalties.append(KL)

                    skip_log_prob = tf.math.log(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=skip_logits, labels=skip)
                    )
                    log_probs.append(tf.reduce_sum(skip_log_prob, keepdims=True))

                    skip = tf.reshape(skip, [1, L])
                    skip = tf.cast(skip, dtype=tf.float32)
                    skip_count.append(tf.reduce_sum(skip))

                    next_input = tf.matmul(skip, tf.concat(anchors, axis=0))
                    next_input /= (1.0 + tf.reduce_sum(skip))
                    next_input = tf.expand_dims(next_input, 1)
                else:
                    next_input = tf.expand_dims(input_arg, 1)

                anchors.append(state_h)
                anchors_W1.append(self.bahdanau_W1(state_h))

            return op_actions, connections, log_probs, skip_penalties

    return Controller(units, batch_size, n_layers)


def run_controller():
    units = 100
    controller_batch_size = 1
    # n_layers = 6
    # out_channels = [8, 16, 32, 64, 128, 256]
    out_channels = [8, 8, 16, 16, 32, 32, 64, 128, 256]
    # out_channels = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256]
    controller = create_controller(units, controller_batch_size, len(out_channels))
    initializer = tf.keras.initializers.GlorotUniform()
    initial_input = initializer(shape=(controller_batch_size, units))
    controller_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0035)

    batch_size = 128
    (xtrain, ytrain), (xtest, ytest) = mg.prepare_data(dataset='cifar10', batch_size=batch_size, shuffle=True)
    pool_size = (2, 2)
    macro, container = mg.build_macro_graph(input_shape=xtrain[0].shape, out_channels=out_channels,
                                            regularizer='l2', regularization=1e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05, decay_steps=10000, decay_rate=0.9, staircase=True)
    # optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=0.9, nesterov=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000,
                                                                 decay_rate=0.9, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    moving_average = 0.1
    exp_ma_window = len(xtrain)
    exp_ma_alpha = 2 / (exp_ma_window + 1)
    # entropy_weight = 0.01  # TODO: paper uses 0.1 but that seems like it would dominate the reward
    op_entropy_weight = 0.1  # TODO: paper uses 0.1 but that seems like it would dominate the reward
    skip_entropy_weight = 0.03  # TODO: paper uses 0.1 but that seems like it would dominate the reward
    skip_weight = 0.2  # TODO: similarly skip weight of 0.8 seems pretty high

    def train_shared(it, moving_average):
        shared_epochs = 1
        batches = len(xtrain)
        # batches = 50
        for e in range(shared_epochs):
            np.random.RandomState(seed=e).shuffle(xtrain)
            np.random.RandomState(seed=e).shuffle(ytrain)
            for m in range(len(xtrain[:batches])):
                print(f'iter{it}, epoch {e}, batch {m}')
                while True:
                    # TODO: rewrite this this is horrible
                    op_actions, connections, log_probs, op_entropys, skip_entropys, skip_count, skip_penaltys = controller(initial_input)
                    op_actions = np.array(op_actions)[:, 0]
                    connect_list = [[1]]
                    connect_list.extend([list(np.array(connection)) for connection in connections])
                    for i in range(1, len(connect_list)):
                        connect_list[i].append(1)
                    if mg.validate_architecture(op_actions, connections, xtrain[m].shape, pool_size):
                        break

                model = mg.build_from_actions(macro, container, op_actions, connect_list)
                with tf.GradientTape() as tape:
                    model_out = model(xtrain[m])
                    loss = loss_fn(ytrain[m], model_out)
                    accuracy_fn.update_state(ytrain[m], model_out)
                    epoch_accuracy_fn.update_state(ytrain[m], model_out)
                    accuracy = accuracy_fn.result()
                    epoch_accuracy = epoch_accuracy_fn.result()
                    accuracy_fn.reset_states()
                    gradients = tape.gradient(loss, model.trainable_variables)
                    moving_average = (exp_ma_alpha * accuracy) + (1 - exp_ma_alpha) * moving_average
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print(
                    f'\nloss={loss}\naccuracy={accuracy}\n'
                    f'epoch accuracy={epoch_accuracy}\nop_actions={op_actions}\nconnections={connect_list}')
        return moving_average

    def train_controller(it, moving_average):
        np.random.RandomState(seed=it).shuffle(xtrain)
        np.random.RandomState(seed=it).shuffle(ytrain)
        steps = len(xtrain)
        # steps = 10

        steps_per_batch = 30
        # steps_per_batch = 3

        n_episdoes = steps // steps_per_batch

        for episode in range(n_episdoes):
            batch_gradients = [tf.zeros_like(controller.trainable_variables[i]) for i in
                               range(len(controller.trainable_variables))]

            start_step = steps_per_batch * episode

            for step in range(start_step, start_step + steps_per_batch):
                print(f'iter{it}, step {step}')
                # while True:
                # TODO: rewrite this this is horrible, especially for controller
                with tf.GradientTape() as tape:
                    op_actions, connections, log_probs, op_entropys, skip_entropys, skip_count, skip_penaltys = controller(initial_input)

                    op_actions = np.array(op_actions)[:, 0]
                    # op_actions = [4] * 12
                    connect_list = [[1]]
                    connect_list.extend([list(np.array(connection)) for connection in connections])
                    for i in range(1, len(connect_list)):
                        connect_list[i].append(1)
                    if not mg.validate_architecture(op_actions, connect_list, xtrain[step].shape, pool_size):
                        accuracy_fn.update_state(np.ones((128, 1)), np.zeros(
                            (128, 1)))  # NOTE: sparse categorical actually takes argmax of y_pred, i.e. 0 for all
                        epoch_accuracy_fn.update_state(np.ones((128, 1)), np.zeros(
                            (128, 1)))  # so just make sure y_true is vector of something that isn't 0s
                        accuracy = accuracy_fn.result()
                        epoch_accuracy = epoch_accuracy_fn.result()
                        accuracy_fn.reset_states()
                    else:
                        model = mg.build_from_actions(macro, container, op_actions, connect_list)
                        model_out = model(xtrain[step])
                        accuracy_fn.update_state(ytrain[step], model_out)
                        epoch_accuracy_fn.update_state(ytrain[step], model_out)
                        accuracy = accuracy_fn.result()
                        epoch_accuracy = epoch_accuracy_fn.result()
                        accuracy_fn.reset_states()

                    # update_moving_average(moving_average, accuracy)
                    moving_average = (exp_ma_alpha * accuracy) + (1 - exp_ma_alpha) * moving_average
                    reward = accuracy - moving_average

                    normalize = len(out_channels) * (len(out_channels) - 1) / 2
                    skip_rate = skip_count / normalize  # TOOO: ?
                    reward += op_entropy_weight * op_entropys
                    reward += skip_entropy_weight * skip_entropys

                    sum_probs = tf.reduce_sum([tf.reduce_sum(lp) for lp in log_probs])
                    # loss = tf.constant(0, dtype=tf.float32)
                    # loss = tf.add(loss, tf.multiply(-sum_probs, accuracy))
                    loss = tf.multiply(sum_probs, reward)
                    loss += skip_weight * skip_penaltys
                grads = tape.gradient(loss, controller.trainable_variables)
                batch_gradients = [tf.math.add(g1, g2) for g1, g2 in zip(batch_gradients, grads)]

            controller_optimiser.apply_gradients(zip(batch_gradients, controller.trainable_variables))
            print(
                f'\nloss={loss}\naccuracy={accuracy}\nepoch accuracy={epoch_accuracy}\nreward={reward}\nmv_avg={moving_average}\n'
                f'op_actions={op_actions}\nconnections={connect_list}')
        return moving_average

    def train_controller_old(it):
        steps = len(xtrain)
        for step in range(steps):
            print(f'iter{it}, step {step}')
            # while True:
                # TODO: rewrite this this is horrible, especially for controller
            with tf.GradientTape() as tape:
                op_actions, connections, log_probs, skip_penalties = controller(initial_input)
                sum_probs = tf.reduce_sum([tf.reduce_sum(lp) for lp in log_probs])
                loss = tf.constant(0, dtype=tf.float32)

                op_actions = np.array(op_actions)[:, 0]
                connect_list = [[1]]
                connect_list.extend([list(np.array(connection)) for connection in connections])
                for i in range(1, len(connect_list)):
                    connect_list[i].append(1)
                if not mg.validate_architecture(op_actions, connect_list, xtrain[step].shape, pool_size):
                    continue

                model = mg.build_from_actions(macro, container, op_actions, connect_list)
                model_out = model(xtrain[step])
                accuracy_fn.update_state(ytrain[step], model_out)
                epoch_accuracy_fn.update_state(ytrain[step], model_out)
                accuracy = accuracy_fn.result()
                epoch_accuracy = epoch_accuracy_fn.result()
                accuracy_fn.reset_states()

                loss = tf.add(loss, tf.multiply(-sum_probs, accuracy))
            grads = tape.gradient(loss, controller.trainable_variables)
            controller_optimiser.apply_gradients(zip(grads, controller.trainable_variables))
            print(f'\nloss={loss}\naccuracy={accuracy}\nepoch accuracy={epoch_accuracy}\n'
                  f'op_actions={op_actions}\nconnections={connect_list}')

    # for i in range(100):
    #     epoch_accuracy_fn.reset_states()
    #     train_shared(i)
    #     mg.save_macro(macro, container, filename='train_for_controller.pkl', template='train_for_controller.pkl')

    controller_weights_filename = 'controller_weights/controller_1'
    macro_filename = 'macro_files/macro_1.pkl'
    for i in range(50):
        epoch_accuracy_fn.reset_states()
        moving_average = train_shared(i, moving_average)
        # mg.save_macro(macro, container, filename=macro_filename, template=macro_filename)
        epoch_accuracy_fn.reset_states()
        moving_average = train_controller(i, moving_average)
        # controller.save_weights(filepath=controller_weights_filename, overwrite=True)


def run_controller_only_on_saved_macro(macro_filename, controller_weights_filename):
    units = 100
    controller_batch_size = 1
    # out_channels = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256]
    out_channels = [8, 16, 32, 64, 128, 256]
    controller = create_controller(units, controller_batch_size, len(out_channels))
    # controller.load_weights(controller_weights_filename)
    initializer = tf.keras.initializers.GlorotUniform()
    initial_input = initializer(shape=(controller_batch_size, units))
    controller_optimiser = tf.keras.optimizers.Adam(learning_rate=0.0035)
    batch_size = 128
    (xtrain, ytrain), (xtest, ytest) = mg.prepare_data(dataset='mnist', batch_size=batch_size, shuffle=True)
    pool_size = (2, 2)
    macro, container = mg.restore_macro(macro_filename, xtrain[0].shape, out_channels)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

    moving_average = 0.29
    exp_ma_window = len(xtrain)
    exp_ma_alpha = 2 / (exp_ma_window + 1)
    op_entropy_weight = 0.01  # TODO: paper uses 0.1 but that seems like it would dominate the reward
    skip_entropy_weight = 0.01
    skip_weight = 0.2  # TODO: similarly skip weight of 0.8 seems pretty high

    # def update_moving_average(old_avg, new_val):
    #     moving_average = (exp_ma_alpha * new_val) + (1 - exp_ma_alpha) * old_avg


    def train_controller(it, moving_average):
        np.random.RandomState(seed=it).shuffle(xtrain)
        np.random.RandomState(seed=it).shuffle(ytrain)
        steps = len(xtrain)
        # steps = 10

        steps_per_batch = 10

        n_episdoes = steps // steps_per_batch



        for episode in range(n_episdoes):
            batch_gradients = [tf.zeros_like(controller.trainable_variables[i]) for i in
                               range(len(controller.trainable_variables))]

            start_step = steps_per_batch * episode

            for step in range(start_step, start_step + steps_per_batch):
                print(f'iter{it}, step {step}')
                # while True:
                    # TODO: rewrite this this is horrible, especially for controller
                with tf.GradientTape() as tape:
                    op_actions, connections, log_probs, op_entropys, skip_entropys, skip_count, skip_penaltys = controller(initial_input)

                    op_actions = np.array(op_actions)[:, 0]
                    # op_actions = [4] * 12
                    connect_list = [[1]]
                    connect_list.extend([list(np.array(connection)) for connection in connections])
                    for i in range(1, len(connect_list)):
                        connect_list[i].append(1)
                    if not mg.validate_architecture(op_actions, connect_list, xtrain[step].shape, pool_size):
                        accuracy_fn.update_state(np.ones((128, 1)), np.zeros((128, 1)))  # NOTE: sparse categorical actually takes argmax of y_pred, i.e. 0 for all
                        epoch_accuracy_fn.update_state(np.ones((128, 1)), np.zeros((128, 1)))  # so just make sure y_true is vector of something that isn't 0s
                        accuracy = accuracy_fn.result()
                        epoch_accuracy = epoch_accuracy_fn.result()
                        accuracy_fn.reset_states()
                    else:
                        model = mg.build_from_actions(macro, container, op_actions, connect_list)
                        model_out = model(xtrain[step])
                        accuracy_fn.update_state(ytrain[step], model_out)
                        epoch_accuracy_fn.update_state(ytrain[step], model_out)
                        accuracy = accuracy_fn.result()
                        epoch_accuracy = epoch_accuracy_fn.result()
                        accuracy_fn.reset_states()

                    # update_moving_average(moving_average, accuracy)
                    moving_average = (exp_ma_alpha * accuracy) + (1 - exp_ma_alpha) * moving_average
                    reward = accuracy - moving_average

                    normalize = len(out_channels) * (len(out_channels) - 1) / 2
                    skip_rate = skip_count / normalize  # TOOO: ?
                    reward += op_entropy_weight * op_entropys
                    reward += skip_entropy_weight * skip_entropys


                    sum_probs = tf.reduce_sum([tf.reduce_sum(lp) for lp in log_probs])
                    # loss = tf.constant(0, dtype=tf.float32)
                    # loss = tf.add(loss, tf.multiply(-sum_probs, accuracy))
                    loss = tf.multiply(sum_probs, reward)
                    loss += skip_weight * skip_penaltys
                grads = tape.gradient(loss, controller.trainable_variables)
                batch_gradients = [tf.math.add(g1, g2) for g1, g2 in zip(batch_gradients, grads)]

            controller_optimiser.apply_gradients(zip(batch_gradients, controller.trainable_variables))
            print(f'\nloss={loss}\naccuracy={accuracy}\nepoch accuracy={epoch_accuracy}\nreward={reward}\nmv_avg={moving_average}\n'
                  f'op_actions={op_actions}\nconnections={connect_list}')
        return moving_average

    for i in range(100):
        moving_average = train_controller(i, moving_average)
        controller.save_weights(filepath=controller_weights_filename, overwrite=True)



if __name__ == '__main__':
    run_controller()
    # macro_filename = 'train_for_controller.pkl'
    # controller_weights_filename = 'controller_weights/controller_0/'
    # run_controller_only_on_saved_macro(macro_filename, controller_weights_filename)
