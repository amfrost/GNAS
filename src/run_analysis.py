import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import static_gnn_gnas as gnas
from static_gnn_gnas import GNAS


class RunAnalysis(object):
    def __init__(self, run_id):
        super(RunAnalysis, self).__init__()
        self.run_id = run_id
        self.run_dir = f'../output/{run_id}/'
        self.analysis_dir = self.run_dir + 'analysis/'

        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)


    def plot_macro_loss_vs_learning_rate(self, epochs=1):
        log = pickle.load(open(self.run_dir + 'log.pkl', 'rb'))
        macro_batch_losses = log['macro_batch_losses']
        macro_batch_accs = log['macro_batch_accs']
        losses = macro_batch_losses[:epochs]
        accs = macro_batch_accs[:epochs]
        losses = [np.array(loss) for loss in losses]
        accs = [np.array(acc) for acc in accs]
        losses = np.concatenate(losses)
        accs = np.concatenate(accs)
        loss_series = pd.Series(losses)
        acc_series = pd.Series(accs)
        ema = loss_series.ewm(span=250).mean()
        emacc = acc_series.ewm(span=250).mean()
        plt.figure()
        plt.plot(ema)
        # plt.ylim([2.0, 2.5])
        plt.savefig(self.analysis_dir + f'loss_vs_learning_rate_{epochs}_rounds.png')
        plt.figure()
        plt.plot(emacc)
        # plt.ylim([2.0, 2.5])
        plt.savefig(self.analysis_dir + f'acc_vs_learning_rate_{epochs}_rounds.png')
        print('test')

    def plot_training_accuracy_per_epoch(self, to_epoch=70):
        log = pickle.load(open(self.run_dir + 'log.pkl', 'rb'))
        macro_accs = log['macro_iteration_reported_acc'][:to_epoch]
        ctrl_accs = log['ctrl_iteration_reported_acc'][:to_epoch]
        macro_iteration_learn_rate = log['macro_iteration_learn_rate'][:to_epoch]

        epochs = [i for i in range(70)]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, macro_accs, label='Train accuracy (graph)')
        plt.plot(epochs, ctrl_accs, label='Validation accuracy (controller)')
        plt.xlabel('Iteration')
        plt.ylabel('Correct classification rate')
        plt.legend()
        plt.savefig(self.analysis_dir + f'acc_per_epoch_{to_epoch}.png')

        fig = plt.figure(figsize=(3, 4))
        ax = plt.plot(epochs, macro_iteration_learn_rate)
        fig.axes[0].yaxis.set_label_position('right')
        fig.axes[0].yaxis.set_ticks_position('right')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        # plt.tick_params(axis='y', labelleft='off', labelright='on')
        plt.savefig(self.analysis_dir + f'learn_rate_{to_epoch}.png', bbox_inches='tight')



    def plot_ctrl_loss_vs_learning_rate(self, iters=1, override_iters_from=0):
        log = pickle.load(open(self.run_dir + 'log.pkl', 'rb'))
        macro_batch_losses = log['macro_batch_losses']
        ctrl_batch_architectures = log['ctrl_batch_architectures']
        ctrl_batch_losses = log['ctrl_batch_losses']
        ctrl_batch_accs = log['ctrl_batch_accs']
        losses = ctrl_batch_losses[override_iters_from:override_iters_from + iters]
        accs = ctrl_batch_accs[override_iters_from: override_iters_from + iters]
        losses = [np.array(loss) for loss in losses]
        accs = [np.array(acc) for acc in accs]
        # losses = np.concatenate([[np.mean(losses[j][i]) for i in range(losses[j].shape[0])] for j in range(iters)])
        accs = np.concatenate([[np.mean(accs[j][i]) for i in range(accs[j].shape[0])] for j in range(iters)])
        # loss_series = pd.Series(losses)
        acc_series = pd.Series(accs)
        # ema = loss_series.ewm(span=1000).mean()
        emacc = acc_series.ewm(span=1000).mean()
        # plt.figure()
        # plt.plot(ema)
        # # plt.ylim([-0.2, 0.2])
        # plt.savefig(self.analysis_dir + f'loss_vs_learning_rate_{iters}_rounds.png')
        plt.figure()
        plt.plot(emacc)
        # plt.ylim([2.0, 2.5])
        plt.savefig(self.analysis_dir + f'acc_vs_learning_rate_{iters}_rounds.png')
        print('test')

    def plot_architecture_distribution(self, epoch=150, batch_size=100):
        log = pickle.load(open(self.run_dir + 'log.pkl', 'rb'))

        architectures = [arc[0][0] for arc in log['ctrl_batch_architectures'][epoch]]
        op_counts = np.zeros((6, 6))
        for arc in architectures:
            for i in range(len(arc)):
                op_counts[arc[i], i] += 1

        op_counts = np.array([i/batch_size for i in op_counts])
        fig, ax = plt.subplots()
        im = ax.imshow(op_counts, cmap='PuBu', vmin=0.0, vmax=1.0)


        for i in range(6):
            for j in range(6):
                ax.text(j, i, round(op_counts[i, j], 2), ha='center', va='center', color='darkslateblue')

        ylabels = ['0. conv3x3', '1. conv5x5', '2. sep3x3', '3. sep5x5', '4. avgpool', '5. maxpool']
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        plt.xlabel('Layer')

        # plt.colorbar(ax=ax, cmap='PuBu', values=[0.0, 1.0], orientation='vertical')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('$p(x)$', rotation=0)

        plt.savefig(self.run_dir + f'/analysis/ops_epoch{epoch}.png')

        skip_counts = np.zeros((6, 6))
        for skips in [arc[0][1] for arc in log['ctrl_batch_architectures'][epoch]]:
            for i in range(6):
                for j in range(i):
                    skip_counts[i][j] += skips[i][j]

        skip_counts = np.array([i/batch_size for i in skip_counts])
        fig, ax = plt.subplots()
        im = ax.imshow(skip_counts, cmap='PuBu', vmin=0.0, vmax=1.0)

        for i in range(6):
            for j in range(6):
                ax.text(j, i, round(skip_counts[i, j], 2) if j < i else '', ha='center', va='center', color='darkslateblue')

        # ylabels = ['conv3x3', 'conv5x5', 'sep3x3', 'sep5x5', 'avgpool', 'maxpool']
        # ax.set_yticks(np.arange(len(ylabels)))
        # ax.set_yticklabels(ylabels)
        # plt.xlabel('Layer')
        ylbl = plt.ylabel('To layer', rotation=0, labelpad=30)
        # ylbl.set_rotation(0)
        plt.xlabel('From layer')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('$p(x)$', rotation=0)

        plt.savefig(self.run_dir + f'/analysis/skips_epoch{epoch}.png')


        print('test')


if __name__ == '__main__':
    ra = RunAnalysis(run_id='47-nognn')
    # ra.plot_architecture_distribution(epoch=0)
    # ra.plot_architecture_distribution(epoch=1)
    # ra.plot_architecture_distribution(epoch=2)
    # ra.plot_architecture_distribution(epoch=3)
    # ra.plot_architecture_distribution(epoch=4)
    # ra.plot_architecture_distribution(epoch=5)
    # ra.plot_architecture_distribution(epoch=10)
    # ra.plot_architecture_distribution(epoch=20)
    # ra.plot_architecture_distribution(epoch=29)
    # ra.plot_architecture_distribution(epoch=69)
    # ra.plot_architecture_distribution(epoch=149)
    # ra.plot_macro_loss_vs_learning_rate(epochs=5)
    ra.plot_training_accuracy_per_epoch(to_epoch=70)
    # ra.plot_ctrl_loss_vs_learning_rate(iters=8, override_iters_from=9)
