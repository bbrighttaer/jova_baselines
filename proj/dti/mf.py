# Author: bbrighttaer
# Project: jova
# Date: 7/2/19
# Time: 1:24 PM
# File: mf.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import json
import os
import pickle
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
from soek import *
from soek.bopt import GPMinArgs

import jova.metrics as mt
from jova import cuda
from jova.data import get_data, load_proteins
from jova.metrics import compute_model_performance
from jova.nn.models import MatrixFactorization
from jova.utils import Trainer
from jova.utils.math import ExpAverage
from jova.utils.train_helpers import parse_hparams

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

if cuda:
    torch.cuda.set_device(0)

seeds = [1, 8, 64]


class MF(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset):
        M, all_comps, all_prots = train_dataset['labels'], train_dataset['compounds'], train_dataset['proteins']
        model = MatrixFactorization(len(all_comps), len(all_prots), k=hparams['latent_dim'])
        if cuda:
            model = model.cuda()

        # optimizer configuration
        optimizer = {
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
            "asgd": torch.optim.ASGD,
            "rmsprop": torch.optim.RMSprop,
            "Rprop": torch.optim.Rprop,
            "sgd": torch.optim.SGD,
        }.get(hparams["optimizer"].lower(), None)
        assert optimizer is not None, "{} optimizer could not be found"

        # filter optimizer arguments
        optim_kwargs = dict()
        optim_key = hparams["optimizer"]
        for k, v in hparams.items():
            if "optimizer__" in k:
                attribute_tup = k.split("__")
                if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                    optim_kwargs[attribute_tup[2]] = v
        optimizer = optimizer(model.parameters(), **optim_kwargs)

        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return model, optimizer, all_comps, all_prots, M, metrics

    @staticmethod
    def data_provider(fold, flags, data):
        # Assumes no CV
        return {"train": data[6], "val": None, "test": None}

    @staticmethod
    def evaluate(eval_dict, y, y_pred, w, metrics, tasks, transformers):
        eval_dict.update(compute_model_performance(metrics, y_pred, y, w, transformers, tasks=tasks))
        # scoring
        rms = np.nanmean(eval_dict["nanmean-rms_score"])
        ci = np.nanmean(eval_dict["nanmean-concordance_index"])
        r2 = np.nanmean(eval_dict["nanmean-pearson_r2_score"])
        score = np.nanmean([ci, r2]) - rms
        return score

    @staticmethod
    def train(model, optimizer, comps, prots, M, metrics, transformer, tasks, max_iter=5000, tol=1e-6,
              is_hsearch=False, sim_data_node=None, epoch_ckpt=(100, 10.0)):
        start = time.time()
        best_model_wts = model.state_dict()
        min_error = 10000
        best_epoch = -1
        terminate_training = False
        e_avg = ExpAverage(.01)
        # scheduler = sch.StepLR(optimizer, step_size=500, gamma=0.01)
        criterion = torch.nn.MSELoss()

        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        if sim_data_node:
            sim_data_node.data = [metrics_node]

        if cuda:
            M = M.cuda()

        losses = []
        print('{:<6}\t| {:>10}\t|'.format('Epoch', 'Loss (MSE)'))
        print('-' * 30)
        prev_loss = None
        for epoch in range(max_iter):
            if terminate_training:
                print("Terminating training...")
                break
            model.train()

            Y_hat = model()
            loss = criterion(Y_hat, M)
            print('{:<6}\t| {:>10.4f}\t|'.format(epoch, loss.item()))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            ep_loss = np.nanmean(losses)
            e_avg.update(ep_loss)

            # Adjust the learning rate.
            # scheduler.step()

            if min_error > loss.item():
                min_error = loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if prev_loss is not None and (prev_loss - loss.item()) < tol:
                terminate_training = True
            prev_loss = loss.item()

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        model.load_state_dict(best_model_wts)

        # map drug-target features
        mf_simboost_data_dict = {'comp_mat': model.P.t().numpy(), 'prot_mat': model.Q.t().numpy(),
                                 'comp_index': {c: i for i, c in enumerate(comps)},
                                 'prot_index': {p: j for j, p in enumerate(prots)}}
        return {'model': (model, mf_simboost_data_dict), 'score': -min_error, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model():
        pass


def save_mf_model_and_feats(mf_objs, path, name):
    model, mf_simboost_data_dict = mf_objs
    os.makedirs(path, exist_ok=True)
    # with open(os.path.join(path, "dummy_save.txt"), 'a') as f:
    #     f.write(name + '\n')
    if mf_simboost_data_dict:
        file = os.path.join(path, name + ".mod")
        torch.save(model.state_dict(), file)
        with open(os.path.join(path, name + '_mf_simboost_data_dict.pkl'), 'wb') as f:
            pickle.dump(dict(mf_simboost_data_dict), f)


def main(flags):
    if len(flags["views"]) > 0:
        print("Single views for training: {}, num={}".format(flags["views"], len(flags["views"])))
    else:
        print("No views selected for training")

    for view in flags["views"]:
        dataset_lbl = flags["dataset_name"]
        cview, pview = view
        sim_label = "MF_{}_{}_{}".format(dataset_lbl, cview, pview)
        print(sim_label)

        # Simulation data resource tree
        split_label = flags.split
        # node_label = "{}_{}_{}_{}_{}_{}".format(dataset_lbl, cview, pview, split_label,
        #                                         "eval" if flags["eval"] else "train", date_label)
        node_label = json.dumps({'model_family': 'mf',
                                 'dataset': dataset_lbl,
                                 'cview': cview,
                                 'pview': pview,
                                 'split': split_label,
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'mode': "eval" if flags["eval"] else "train",
                                 'date': date_label
                                 })
        sim_data = DataNode(label=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])

        # For searching over multiple seeds
        hparam_search = None

        for seed in seeds:
            # for data collection of this round of simulation.
            data_node = DataNode(label="seed_%d" % seed)
            nodes_list.append(data_node)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # load data
            print('-------------------------------------')
            print('Running on dataset: %s' % dataset_lbl)
            print('-------------------------------------')

            data_key = {"ecfp4": "MF_ECFP4",
                        "ecfp8": "MF_ECFP8"}.get(cview)
            flags['splitting_alg'] = 'random'
            flags['cv'] = False
            flags['test'] = False
            flags['fold_num'] = 1
            data = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformer = data[2]
            tasks = data[0]
            flags["tasks"] = tasks

            trainer = MF()

            k = 1
            print("{}, {}-{}: Training scheme: train, validation".format(tasks, cview, pview)
                  + (", test split" if flags['test'] else " split"))

            if flags["hparam_search"]:
                print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

                # arguments to callables
                extra_init_args = {}
                extra_data_args = {"flags": flags,
                                   "data": data}
                n_iters = 3000
                extra_train_args = {"transformer": transformer,
                                    "tasks": tasks,
                                    "is_hsearch": True}

                hparams_conf = get_hparam_config(flags)

                if hparam_search is None:
                    search_alg = {"random_search": RandomSearch,
                                  "bayopt_search": BayesianOptSearch}.get(flags["hparam_search_alg"],
                                                                            BayesianOptSearch)
                    search_args = GPMinArgs(n_calls=40, random_state=seed)
                    hparam_search = search_alg(hparam_config=hparams_conf,
                                               num_folds=k,
                                               initializer=trainer.initialize,
                                               data_provider=trainer.data_provider,
                                               train_fn=trainer.train,
                                               save_model_fn=save_mf_model_and_feats,
                                               alg_args=search_args,
                                               init_args=extra_init_args,
                                               data_args=extra_data_args,
                                               train_args=extra_train_args,
                                               data_node=data_node,
                                               split_label=split_label,
                                               sim_label=sim_label,
                                               dataset_label=dataset_lbl,
                                               results_file="{}_{}_dti_{}.csv".format(
                                                   flags["hparam_search_alg"], sim_label, date_label))

                stats = hparam_search.fit(model_dir="models", model_name="".join(tasks))
                print(stats)
                print("Best params = {}".format(stats.best()))
            else:
                invoke_train(trainer, tasks, data, transformer, flags, data_node, sim_label, dataset_lbl)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data, transformer, flags, data_node, view, dataset_lbl):
    try:
        hfile = os.path.join('soek_res', get_hparam_file(dataset_lbl))
        exists = os.path.exists(hfile)
        status = 'Found' if exists else 'Not Found, switching to default hyperparameters'
        print(f'Hyperparameters file:{hfile}, status={status}')
        if not exists:
            raise FileNotFoundError(f'{hfile} not found')
        hyper_params = parse_hparams(hfile)
    except:
        hyper_params = default_hparams_bopt(flags)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data, flags, hyper_params, tasks, trainer, transformer, view)
    else:
        start_fold(data_node, data, flags, hyper_params, tasks, trainer, transformer, view)


def start_fold(sim_data_node, data, flags, hyper_params, tasks, trainer, transformer, view, k=None):
    data = trainer.data_provider(k, flags, data)
    model, optimizer, all_comps, all_prots, pair_to_value_y, metrics = trainer.initialize(hparams=hyper_params,
                                                                                          train_dataset=data["train"],
                                                                                          val_dataset=data["val"],
                                                                                          test_dataset=data["test"])
    if flags["eval"]:
        pass
    else:
        # Train the model
        results = trainer.train(model, optimizer, all_comps, all_prots, pair_to_value_y, metrics, transformer,
                                tasks=tasks, sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = flags.split
        save_mf_model_and_feats(model, flags["model_dir"],
                                "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset_name"], view, flags["model_name"],
                                                               split_label, epoch, score))


def default_hparams_rand(flags):
    return {
        "reg_lambda": 0.1
    }


def default_hparams_bopt(flags):
    return {
        "latent_dim": 84,

        # optimizer params
        "optimizer": "adagrad",
        "optimizer__global__weight_decay": 0.012233128918089326,
        "optimizer__global__lr": 0.21291818884704686,
    }


def get_hparam_config(flags):
    return {
        "latent_dim": DiscreteParam(min=10, max=100),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),
    }


def get_hparam_file(dataset):
    return {'davis': 'bayopt_search_MF_davis_ecfp8_psc_dti_2019_11_27__13_22_48_gp_3000.csv',
            'metz': 'bayopt_search_MF_metz_ecfp8_psc_dti_2019_12_06__17_48_17_gp_3000.csv',
            'kiba': 'bayopt_search_MF_kiba_ecfp8_psc_dti_2019_12_13__01_44_56_gp_3000.csv',
            }.get(dataset.lower(), None)


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kronecker Regularized Least Squares (Pahikkala et al., 2014")

    parser.add_argument("--dataset_name",
                        type=str,
                        default="davis",
                        help="Dataset name.")
    parser.add_argument("--dataset_file",
                        type=str,
                        help="Dataset file.")
    parser.add_argument('--filter_threshold',
                        type=int,
                        default=6,
                        help='Threshold such that entities with observations no more than it would be filtered out.'
                        )
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument('--model_name',
                        type=str,
                        default='model-{}'.format(date_label),
                        help='Directory to store the log files in the training process.'
                        )
    parser.add_argument('--prot_desc_path',
                        action='append',
                        help='A list containing paths to protein descriptors.'
                        )
    parser.add_argument('--no_reload',
                        action="store_false",
                        dest='reload',
                        help='Whether datasets will be reloaded from existing ones or newly constructed.'
                        )
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--prot_view", "-pv",
                        type=str,
                        action="append",
                        help="The view to be simulated. One of [psc, rnn, pcnn]")
    parser.add_argument("--comp_view", "-cv",
                        type=str,
                        action="append",
                        help="The view to be simulated. One of [ecfp4, ecfp8, weave, gconv]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated using CV")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    setattr(flags, "views", [(cv, pv) for cv, pv in zip(args.comp_view, args.prot_view)])
    split = 'warm'
    flags['split'] = split
    flags['predict_cold'] = split == 'cold_drug_target'
    flags['cold_drug'] = split == 'cold_drug'
    flags['cold_target'] = split == 'cold_target'
    flags['cold_drug_cluster'] = split == 'cold_drug_cluster'
    flags['split_warm'] = split == 'warm'
    main(flags)
