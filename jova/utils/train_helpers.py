# Author: bbrighttaer
# Project: jova
# Date: 5/20/19
# Time: 2:47 PM
# File: train_helpers.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from keras.utils.data_utils import get_file
from sklearn import svm as svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from jova import cuda
from jova.utils.math import ExpAverage


def split_mnist(dataset, targets, num_views=2):
    shape = dataset.shape
    view_dim = shape[1] // num_views
    views_data = []

    last_index = 0
    for v in range(num_views):
        v_data = dataset[:, last_index:last_index + view_dim]
        views_data.append((v_data, targets))
        last_index = last_index + view_dim

    return views_data


def train_svm(train_x, train_y):
    print('training SVM...')
    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(train_x, train_y)

    # sanity check
    p = clf.predict(train_x)
    san_acc = accuracy_score(train_y, p)

    return san_acc, clf


def svm_classify(data, C=0.01):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train_label = data[0]
    test_data, test_label = data[1]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label)

    # sanity check
    p = clf.predict(train_data)
    san_acc = accuracy_score(train_label, p)

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    return san_acc, test_acc


def construct_iris_views(dataframe):
    view1 = dataframe[['SepalLengthCm', 'SepalWidthCm', 'label']].values
    view2 = dataframe[['PetalLengthCm', 'PetalWidthCm', 'label']].values
    return view1, view2


def load_data(data_file, url, normalize=True):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    path = get_file(data_file, origin=url)
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_gzip_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    # train_set_x = train_set_x / 255.
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    # valid_set_x = valid_set_x / 255.
    train_set_x = np.vstack([train_set_x, valid_set_x])
    train_set_y = np.vstack([train_set_y.reshape(-1, 1), valid_set_y.reshape(-1, 1)])

    # valid_set_x = valid_set_x / 255.
    test_set_x, test_set_y = make_numpy_array(test_set)
    # test_set_x = test_set_x / 255.

    # Data normalization.
    scaler = StandardScaler()
    train_set_x = scaler.fit_transform(train_set_x)
    test_set_x = scaler.fit_transform(test_set_x)

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def load_gzip_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    # data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_x = np.asarray(data_x, dtype='float32')
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def trim_mnist(vdata, batch_size):
    views = []

    for i, data in enumerate(vdata):
        x_data, y_data = data

        batches = len(x_data) // batch_size

        num_samples = batches * batch_size

        x_data, y_data = x_data[:num_samples, :], y_data[:num_samples]
        views.append((x_data, y_data))

    return views


def process_evaluation_data(dloader, dim, model=None, view_idx=0):
    data_x = None
    data_y = None
    for data in dloader:
        X = data[view_idx][0]
        y = data[view_idx][1]
        if model:
            X = torch.unsqueeze(X, dim=1)
            if cuda:
                X = X.cuda()
                model = model.cuda()
            X = model(X)

            if data_x is None:
                data_x = X.cpu().detach().numpy()
                data_y = y.cpu().detach().numpy()
            else:
                data_x = np.concatenate((data_x, X.cpu().detach().numpy()), axis=0)
                data_y = np.concatenate((data_y, y.cpu().detach().numpy()), axis=0)
    return np.array(data_x).reshape(-1, dim), np.array(data_y).ravel()


def evaluate(model_tr, model_tt, ldim, tr_ldr, tt_ldr, view_idx):
    svm_train_dataset = process_evaluation_data(tr_ldr, ldim, model_tr, view_idx[0])
    svm_test_dataset = process_evaluation_data(tt_ldr, ldim, model_tt, view_idx[1])
    sanity_check, val_accuracy = svm_classify((svm_train_dataset, svm_test_dataset))
    return sanity_check, val_accuracy


def visualize(path, series, xlabel=None, ylabel=None):
    fig = plt.figure()
    legend = []
    for k in series.keys():
        plt.plot(series[k])
        legend.append(k)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)
    plt.savefig(path)
    plt.close(fig)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", world_size=world_size, rank=rank)


def cleanup():
    dist.destroy_process_group()


def run_training(training_fn, nprocs, *args):
    mp.spawn(fn=training_fn,
             args=(nprocs, *args),
             nprocs=nprocs,
             join=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GradStats(object):
    def __init__(self, net, tb_writer=None, beta=.9, bias_cor=False):
        super(GradStats, self).__init__()
        self.net = net
        self.writer = tb_writer
        self._l2 = ExpAverage(beta, bias_cor)
        self._max = ExpAverage(beta, bias_cor)
        self._var = ExpAverage(beta, bias_cor)
        self._window = 1 // (1. - beta)

    @property
    def l2(self):
        return self._l2.value

    @property
    def max(self):
        return self._max.value

    @property
    def var(self):
        return self._var.value

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self._l2.reset()
        self._max.reset()
        self._var.reset()
        self.t = 0

    def stats(self, step_idx=None):
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.net.parameters()
                                if p.grad is not None])
        l2 = np.sqrt(np.mean(np.square(grads)))
        self._l2.update(l2)
        mx = np.max(np.abs(grads))
        self._max.update(mx)
        vr = np.var(grads)
        self._var.update(vr)
        if self.writer:
            assert step_idx is not None, "step_idx cannot be none"
            self.writer.add_scalar("grad_l2", l2, step_idx)
            self.writer.add_scalar("grad_max", mx, step_idx)
            self.writer.add_scalar("grad_var", vr, step_idx)
        return "Grads stats (w={}): L2={}, max={}, var={}".format(int(self._window), self.l2, self.max, self.var)


def get_activation_func(activation):
    from jova.nn.models import NonsatActivation
    return {'relu': torch.nn.ReLU(),
            'leaky_relu': torch.nn.LeakyReLU(.2),
            'sigmoid': torch.nn.Sigmoid(),
            'tanh': torch.nn.Tanh(),
            'softmax': torch.nn.Softmax(),
            'elu': torch.nn.ELU(),
            'nonsat': NonsatActivation()}.get(activation.lower(), torch.nn.ReLU())


class FrozenModels(object):

    def __init__(self):
        self._models = []

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, m):
        self._models = m

    def clear(self):
        self._models.clear()

    def __iter__(self):
        for model in self._models:
            yield model

    def add_model(self, model):
        self._models.append(model)

    def unfreeze(self):
        for model in self._models:
            model.train()

    def freeze(self):
        for model in self._models:
            model.eval()


def create_torch_embeddings(frozen_models_hook, np_embeddings):
    pretrained_embeddings = torch.from_numpy(np_embeddings.astype(np.float)).float()
    # Add zeros as the last row for entries added to embeddings query just to pad them for batch processing.
    padded_embeddings = F.pad(pretrained_embeddings, (0, 0, 0, 1))
    shape = padded_embeddings.shape
    pt_embeddings = torch.nn.Embedding(num_embeddings=shape[0], embedding_dim=shape[1],
                                       _weight=padded_embeddings, padding_idx=shape[0] - 1)
    if frozen_models_hook:
        frozen_models_hook.add_model(pt_embeddings)
    return pt_embeddings


class ViewsReg(object):
    """
    Used to manage all views active in a simulation.
    The format for joint-views arg to be passed is:
    comp1-compN__prot1-protN (note the double underscore)
    For instance, for a combination of the PSC and RNN protein views with ECFP8 and GraphConv views of a compound, the
    argument would be:
    ecfp8-gconv__psc-rnn
    """
    # targets
    pcnn_views = ['pcnn', 'pcnn2d']
    embedding_based_views = ['rnn', 'p2v'] + pcnn_views
    all_prot_views = ['psc'] + embedding_based_views

    # compounds
    graph_based_views = ['weave', 'gconv', 'gnn']
    all_comp_views = ['ecfp8'] + graph_based_views

    def __init__(self):
        self.c_views = []
        self.p_views = []
        self.feat_dict = {'ecfp8': 'ECFP8', 'weave': 'Weave', 'gconv': 'GraphConv', 'gnn': 'GNN'}

    def parse_views(self, jova_arg):
        """
        Sets the active protein and compound views of the registry

        :param jova_arg: str
            A set of view combinations for model training/simulation.
        """
        assert isinstance(jova_arg, str)
        self.c_views, self.p_views = [[v for v in seg.split('-')] for seg in jova_arg.split('__')]
        for v in self.c_views + self.p_views:
            all = self.all_comp_views + self.all_prot_views
            assert (v in all), "{} not in {}".format(v, str(all))


def parse_hparams(file):
    hdata = pd.read_csv(file, header=0, nrows=1)
    hdict = hdata.to_dict('index')
    hdict = hdict[list(hdict.keys())[0]]
    return _rec_parse_hparams(hdict)


def _rec_parse_hparams(p_dict):
    hparams = {}
    for k in p_dict:
        v = p_dict[k]
        try:
            new_v = eval(v) if isinstance(v, str) else v
            if isinstance(new_v, dict):
                new_v = _rec_parse_hparams(new_v)
            hparams[k] = new_v
        except NameError:
            hparams[k] = v
    return hparams


def np_to_plot_data(y):
    y = y.squeeze()
    if y.shape == ():
        return [float(y)]
    else:
        return y.squeeze().tolist()