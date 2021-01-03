import tensorflow as tf
import numpy as np
import macro_graph as mg
import pickle
import os

DATA_DIR = '../data/mnist'


def get_dataset(dataset='mnist'):
    if dataset == 'mnist':
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
        xtrain = xtrain[:, :, :, np.newaxis]
        xtest = xtest[:, :, :, np.newaxis]
    elif dataset == 'fashion_mnist':
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()
        xtrain = xtrain[:, :, :, np.newaxis]
        xtest = xtest[:, :, :, np.newaxis]
    elif dataset == 'cifar10':
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError("Dataset must be in ['mnist', 'fashion_mnist', 'cifar10'].")

    return (xtrain, ytrain), (xtest, ytest)


def normalise_xdata(xtrain, xtest=None):
    mu = np.mean(xtrain)
    sigma = np.std(xtrain)
    xtrain = (xtrain - mu) / sigma
    if xtest is not None:
        xtest = (xtest - mu) / sigma

    return xtrain, xtest


def shuffle_xy(x, y, seed=None):
    if seed is None:
        seed = np.random.randint(10000)
    np.random.RandomState(seed).shuffle(x)
    np.random.RandomState(seed).shuffle(y)
    return x, y


def batch_xy(x, y, batch_size=128, drop_remainder=True, shuffle_batches=True, seed=None):
    if not drop_remainder:
        raise NotImplementedError("Only implemented for drop_remainder=True")
    if seed is None:
        seed = np.random.randint(10000)

    n_batches = len(x) // batch_size
    remainder = len(x) % batch_size
    x_batches = np.split(x[:-remainder], n_batches, axis=0)
    y_batches = np.split(y[:-remainder], n_batches)

    if shuffle_batches:
        np.random.RandomState(seed).shuffle(x_batches)
        np.random.RandomState(seed).shuffle(y_batches)

    return x_batches, y_batches


def prepare_data(dataset='mnist', batch_size=128, shuffle=True, seed=None, drop_remainder=True,
                 shuffle_batches=True, return_test=False, batch_test=False, valid_size=0.2):
    if seed is None:
        seed = np.random.randint(10000)

    (xtrain, ytrain), (xtest, ytest) = get_dataset(dataset)
    xtrain, xtest = normalise_xdata(xtrain, xtest)
    if shuffle:
        xtrain, ytrain = shuffle_xy(xtrain, ytrain, seed=seed)
        xtest, ytest = shuffle_xy(xtest, ytest, seed=seed)

    if valid_size is not None:
        xtrain, ytrain, xvalid, yvalid = split_xy_train_valid(xtrain, ytrain, valid_size,
                                                              shuffle_first=True, seed=seed)
        xvalid_batches, yvalid_batches = batch_xy(xvalid, yvalid, batch_size=batch_size, drop_remainder=drop_remainder,
                                                  shuffle_batches=shuffle_batches, seed=seed)


    xtrain_batches, ytrain_batches = batch_xy(xtrain, ytrain, batch_size=batch_size, drop_remainder=drop_remainder,
                                              shuffle_batches=shuffle_batches, seed=seed)

    return_vals = [xtrain_batches, ytrain_batches]
    if valid_size is not None:
        return_vals.extend([xvalid_batches, yvalid_batches])
    if return_test:
        if batch_test:
            xtest_batches, ytest_batches = batch_xy(xtest, ytest, batch_size=batch_size, drop_remainder=drop_remainder,
                                                    shuffle_batches=shuffle_batches, seed=seed)
            return_vals.extend([xtest_batches, ytest_batches])
        else:
            return_vals.extend([xtest, ytest])

    return tuple(return_vals)

def split_xy_train_valid(x, y, valid_size, shuffle_first=True, seed=None):
    if seed is None:
        seed = np.random.randint(10000)
    if shuffle_first:
        x, y = shuffle_xy(x, y, seed)

    split_index = int(valid_size * len(x))

    xtrain, xvalid, ytrain, yvalid = x[split_index:], x[:split_index], y[split_index:], y[:split_index]
    return xtrain, ytrain, xvalid, yvalid




def save_macro(macro_graph, container, filename, template=None):
    if template is not None and os.path.exists(template):
        with open(template, 'rb') as f:
            template_pkl, template_container_shape, template_container_weights = pickle.load(f)
    else:
        template_pkl, template_container_shape, template_container_weights = None, None, None
    macro_pkl = [None] * len(macro_graph)
    for i in range(len(macro_graph)):
        layer = macro_graph[i]
        layer_weights = [None] * len(layer)
        for j in range(len(layer)):
            ops_node = layer[j]
            ops = ops_node['ops']
            ops_weights = [None] * len(ops)
            for k in range(len(ops)):
                if ops[k].weights:
                    ops_k_weights = {}
                    ops_k_weights['weights'] = ops[k].get_weights()
                    if template_pkl is not None:
                        ops_k_weights['shape'] = template_pkl[i][j][k]['shape']
                    else:
                        ops_k_weights['shape'] = ops[k].input.shape
                    ops_weights[k] = ops_k_weights
            layer_weights[j] = ops_weights
        macro_pkl[i] = layer_weights

    if template_container_shape is not None:
        container_fc_shape = template_container_shape
    else:
        container_fc_shape = container['fc'].input.shape
    container_fc_weights = container['fc'].get_weights()

    with open(filename, 'wb') as f:
        pickle.dump((macro_pkl, container_fc_shape, container_fc_weights), f)


def restore_macro(filename, input_shape, output_depths):
    macrograph = mg.MacroGraph(input_shape=input_shape, output_depths=output_depths)
    new_graph_layers, new_containing_comps = macrograph.graph_layers, macrograph.containing_components
    with open(filename, 'rb') as f:
        macro_pkl, container_fc_shape, container_fc_weights = pickle.load(f)
    for i in range(len(new_graph_layers)):
        layer = new_graph_layers[i]
        for j in range(len(layer)):
            ops_node = layer[j]
            ops = ops_node['ops']
            for k in range(len(ops)):
                if macro_pkl[i][j][k]:
                    new_graph_layers[i][j]['ops'][k].build(input_shape=macro_pkl[i][j][k]['shape'])
                    new_graph_layers[i][j]['ops'][k].set_weights(macro_pkl[i][j][k]['weights'])
    new_containing_comps['fc'].build(input_shape=container_fc_shape)
    new_containing_comps['fc'].set_weights(container_fc_weights)
    return new_graph_layers, new_containing_comps
