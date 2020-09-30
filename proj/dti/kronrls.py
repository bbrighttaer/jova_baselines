# Author: bbrighttaer
# Project: jova
# Date: 7/2/19
# Time: 1:24 PM
# File: kronrls.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from soek import *
from soek.bopt import GPMinArgs

import jova.metrics as mt
from jova.data import get_data, load_proteins
from jova.data.data import Pair
from jova.metrics import compute_model_performance
from jova.trans import undo_transforms
from jova.utils import Trainer
from jova.utils.io import save_numpy_array, load_numpy_array

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1, 8, 64]


class KronRLS(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, kernel_data):
        data = {"train": train_dataset, "val": val_dataset, "test": test_dataset, "kernel_data": kernel_data}
        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return data, hparams['reg_lambda'], metrics

    @staticmethod
    def data_provider(fold, flags, data):
        if not flags['cv']:
            print("Training scheme: train, validation" + (", test split" if flags['test'] else " split"))
            train_dataset = (data[1][0].X, data[1][0].y, data[1][0].w)
            valid_dataset = (data[1][1].X, data[1][1].y, data[1][1].w)
            test_dataset = None
            if flags['test']:
                test_dataset = (data[1][2].X, data[1][2].y, data[1][2].w)
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset, "kernel_data": data[1][3]}
        else:
            train_dataset = (data[1][fold][0].X, data[1][fold][0].y, data[1][fold][0].w)
            valid_dataset = (data[1][fold][1].X, data[1][fold][1].y, data[1][fold][1].w)
            test_dataset = (data[1][fold][2].X, data[1][fold][2].y, data[1][fold][2].w)
            kernel_data = data[1][fold][3]
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset, "kernel_data": kernel_data}
        return data

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
    def train(data, reg_lambda, metrics, transformer, drug_kernel_dict, prot_kernel_dict, tasks, sim_data_node,
              is_hsearch=False):
        start = time.time()
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        eval_scores_lst = []
        eval_scores_node = DataNode(label="validation_score", data=eval_scores_lst)
        eval_predicted_vals = []
        eval_true_vals = []
        eval_model_preds_node = DataNode(label="model_predictions",
                                         data={"y": eval_true_vals, "y_pred": eval_predicted_vals})
        if sim_data_node:
            sim_data_node.data = [metrics_node, eval_scores_node, eval_model_preds_node]

        kernel_data = data['kernel_data']
        KD = kernel_data['KD']
        KT = kernel_data['KT']
        Y = kernel_data['Y']
        W = kernel_data['W']

        # Eigen decompositions
        Lambda, V = np.linalg.eigh(KD)
        Lambda = Lambda.reshape((-1, 1))
        Sigma, U = np.linalg.eigh(KT)
        Sigma = Sigma.reshape((-1, 1))

        # Compute C
        newevals = 1. / (Lambda @ Sigma.T + reg_lambda)
        # newevals = newevals.T
        VTYU = V.T @ Y @ U
        C = np.multiply(VTYU, newevals)

        # compute weights
        A = V @ C @ U.T

        # training loss
        P_train = KD @ A @ KT.T
        tr_loss = mean_squared_error(Y.reshape(-1, 1), P_train.reshape(-1, 1), W.reshape(-1, 1))

        # Evaluation
        print('Eval mat construction started')
        if is_hsearch:
            KD_eval = kernel_data['KD_val']
            KT_eval = kernel_data['KT_val']
            Y_eval = kernel_data['Y_val']
            W_eval = kernel_data['W_val']
        else:
            KD_eval = kernel_data['KD_test']
            KT_eval = kernel_data['KT_test']
            Y_eval = kernel_data['Y_test']
            W_eval = kernel_data['W_test']
        P_val = KD_eval @ A @ KT_eval.T
        y_hat = P_val.reshape(-1, 1)
        y = Y_eval.reshape(-1, 1)
        w = W_eval.reshape(-1, 1)
        eval_loss = mean_squared_error(y, y_hat, w)

        # Metrics
        eval_dict = {}
        score = KronRLS.evaluate(eval_dict, y, y_hat, w, metrics, tasks, transformer)
        for m in eval_dict:
            if m in metrics_dict:
                metrics_dict[m].append(eval_dict[m])
            else:
                metrics_dict[m] = [eval_dict[m]]
        print(f'Training loss={tr_loss}, evaluation loss={eval_loss}, score={score}, metrics={str(eval_dict)}')

        # apply transformers
        y_hat = y_hat[w.nonzero()[0]]
        y = y[w.nonzero()[0]]
        eval_predicted_vals.extend(undo_transforms(y_hat, transformer).squeeze().tolist())
        eval_true_vals.extend(undo_transforms(y, transformer).squeeze().tolist())
        eval_scores_lst.append(score)
        print(f'eval loss={eval_loss}, score={score}, metrics={str(eval_dict)}')

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        return {'model': A, 'score': score, 'epoch': 0}

    @staticmethod
    def evaluate_model(data, model_dir, model_file, metrics, transformer, drug_kernel_dict, prot_kernel_dict, tasks,
                       sim_data_node):
        print("Model evaluation...")
        start = time.time()
        eval_metrics_dict = {}
        eval_metrics_node = DataNode(label="validation_metrics", data=eval_metrics_dict)
        eval_scores_lst = []
        eval_scores_node = DataNode(label="validation_score", data=eval_scores_lst)
        eval_predicted_vals = []
        eval_true_vals = []
        eval_model_preds_node = DataNode(label="model_predictions",
                                         data={"y": eval_true_vals, "y_pred": eval_predicted_vals})
        if sim_data_node:
            sim_data_node.data = [eval_metrics_node, eval_scores_node, eval_model_preds_node]

        kernel_data = data['kernel_data']

        # compute weights
        A = load_numpy_array(os.path.join(model_dir, model_file))

        # Test
        KD_eval = kernel_data['KD_test']
        KT_eval = kernel_data['KT_test']
        Y_eval = kernel_data['Y_test']
        W_eval = kernel_data['W_test']

        P_val = KD_eval @ A @ KT_eval.T
        y_hat = P_val.reshape(-1, 1)
        y = Y_eval.reshape(-1, 1)
        w = W_eval.reshape(-1, 1)
        eval_loss = mean_squared_error(y, y_hat, w)

        # Metrics
        eval_dict = {}
        score = KronRLS.evaluate(eval_dict, y, y_hat, w, metrics, tasks, transformer)
        for m in eval_dict:
            if m in eval_metrics_dict:
                eval_metrics_dict[m].append(eval_dict[m])
            else:
                eval_metrics_dict[m] = [eval_dict[m]]
        # apply transformers

        y_hat = y_hat[w.nonzero()[0]]
        y = y[w.nonzero()[0]]
        eval_predicted_vals.extend(undo_transforms(y_hat, transformer).squeeze().tolist())
        eval_true_vals.extend(undo_transforms(y, transformer).squeeze().tolist())
        eval_scores_lst.append(score)
        print(f'eval loss={eval_loss}, score={score}, metrics={str(eval_dict)}')

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def compute_eval_mat(data, q, idx, Kd_dict, Kt_dict, A_dict):
    rows = []
    for x_i in data:
        mol, prot = x_i
        row = [Kd_dict[Pair(mol, pair.p1)] * Kt_dict[Pair(prot, pair.p2)] for pair in A_dict]
        rows.append(row)
    q.append((idx, rows))


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def to_tensor(array):
    return torch.from_numpy(array)


def main(flags):
    if len(flags["views"]) > 0:
        print("Single views for training: {}, num={}".format(flags["views"], len(flags["views"])))
    else:
        print("No views selected for training")

    for view in flags["views"]:
        cview, pview = view
        sim_label = "KronRLS_{}_{}".format(cview, pview)
        print(sim_label)

        # Simulation data resource tree
        split_label = flags['split']
        dataset_lbl = flags["dataset_name"]
        # node_label = "{}_{}_{}_{}_{}".format(dataset_lbl, sim_label, split_label,
        #                                      "eval" if flags["eval"] else "train", date_label)
        node_label = json.dumps({'model_family': 'kronrls',
                                 'dataset': dataset_lbl,
                                 'mode': "eval" if flags["eval"] else "train",
                                 'split': split_label,
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'date': date_label})
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

            # load data
            print('-------------------------------------')
            print('Running on dataset: %s' % dataset_lbl)
            print('-------------------------------------')

            data_key = {"ecfp4": "KRLS_ECFP4",
                        "ecfp8": "KRLS_ECFP8"}.get(cview)
            data = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformer = data[2]
            drug_kernel_dict, prot_kernel_dict = data[4]
            tasks = data[0]
            flags["tasks"] = tasks

            trainer = KronRLS()

            if flags["cv"]:
                k = flags["fold_num"]
                print("{}, {}-{}: Training scheme: {}-fold cross-validation".format(tasks, cview, pview, k))
            else:
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
                                    "drug_kernel_dict": drug_kernel_dict,
                                    "prot_kernel_dict": prot_kernel_dict,
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
                                               save_model_fn=save_dummy,
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
                invoke_train(trainer, tasks, data, transformer, flags, data_node, sim_label, drug_kernel_dict,
                             prot_kernel_dict)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data, transformer, flags, data_node, view, drug_kernel_dict, prot_kernel_dict):
    hyper_params = default_hparams_bopt(flags)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
                       prot_kernel_dict, k)
    else:
        start_fold(data_node, data, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
                   prot_kernel_dict)


def start_fold(sim_data_node, data, flags, hyper_params, tasks, trainer, transformer, view, drug_kernel_dict,
               prot_kernel_dict, k=None):
    data = trainer.data_provider(k, flags, data)
    _data, reg_lambda, metrics = trainer.initialize(hparams=hyper_params,
                                                    train_dataset=data["train"],
                                                    val_dataset=data["val"],
                                                    test_dataset=data["test"],
                                                    kernel_data=data['kernel_data'])
    if flags["eval"]:
        trainer.evaluate_model(data, flags.model_dir, flags.eval_model_name, metrics, transformer,
                               drug_kernel_dict, prot_kernel_dict, tasks, sim_data_node)
    else:
        # Train the model
        results = trainer.train(data, reg_lambda, metrics, transformer, drug_kernel_dict, prot_kernel_dict,
                                tasks=tasks, sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        save_numpy_array(model, flags["model_dir"],
                         "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset_name"], view, flags["model_name"],
                                                        flags.split, epoch, score))


def default_hparams_rand(flags):
    return {
        "reg_lambda": 1.0
    }


def default_hparams_bopt(flags):
    return {
        "reg_lambda": 1.0
    }


def get_hparam_config(flags):
    return {
        "reg_lambda": LogRealParam()
    }


def save_dummy(array, path, name):
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name)
    with open(os.path.join(path, "dummy_save_kronrls.txt"), 'a') as f:
        f.write(name + '\n')


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

    # Either CV or standard train-val(-test) split.
    scheme = parser.add_mutually_exclusive_group()
    scheme.add_argument("--fold_num",
                        default=-1,
                        type=int,
                        choices=range(3, 11),
                        help="Number of folds for cross-validation")
    scheme.add_argument("--test",
                        action="store_true",
                        help="Whether a test set should be included in the data split")

    parser.add_argument("--splitting_alg",
                        choices=["random", "scaffold", "butina", "index", "task"],
                        default="random",
                        type=str,
                        help="Data splitting algorithm to use.")
    parser.add_argument('--filter_threshold',
                        type=int,
                        default=6,
                        help='Threshold such that entities with observations no more than it would be filtered out.'
                        )
    parser.add_argument('--split',
                        help='Splitting scheme to use. Options are: [warm, cold_drug, cold_target, cold_drug_target]',
                        action='append',
                        type=str,
                        dest='split_schemes'
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
    parser.add_argument('--mp', '-mp', action='store_true', help="Multiprocessing option")

    args = parser.parse_args()
    procs = []
    use_mp = args.mp
    for split in args.split_schemes:
        flags = Flags()
        args_dict = args.__dict__
        for arg in args_dict:
            setattr(flags, arg, args_dict[arg])
        setattr(flags, "cv", True if flags.fold_num > 2 else False)
        setattr(flags, "views", [(cv, pv) for cv, pv in zip(args.comp_view, args.prot_view)])
        flags['split'] = split
        flags['predict_cold'] = split == 'cold_drug_target'
        flags['cold_drug'] = split == 'cold_drug'
        flags['cold_target'] = split == 'cold_target'
        flags['cold_drug_cluster'] = split == 'cold_drug_cluster'
        flags['split_warm'] = split == 'warm'
        if use_mp:
            p = mp.Process(target=main, args=(flags,))
            procs.append(p)
            p.start()
        else:
            main(flags)
    for proc in procs:
        proc.join()
