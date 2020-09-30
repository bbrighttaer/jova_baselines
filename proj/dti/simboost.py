# Author: bbrighttaer
# Project: jova
# Date: 7/2/19
# Time: 1:24 PM
# File: simboost.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import pickle
import random
import time
from datetime import datetime as dt

import numpy as np
import xgboost as xgb
from soek import *

import jova.metrics as mt
from jova.data import get_data, load_proteins
from jova.data.data import Pair
from jova.metrics import compute_model_performance
from jova.trans import undo_transforms
from jova.utils import Trainer
from jova.utils.io import load_dict_model, load_pickle

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

# seeds = [123, 124, 125]
seeds = [1, 8, 64]


class SimBoost(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset):
        params = {'objective': hparams['objective'], 'max_depth': hparams['max_depth'],
                  'subsample': hparams['subsample'], 'colsample_bytree': hparams['colsample_bytree'],
                  'n_estimators': hparams['n_estimators'], 'gamma': hparams['gamma'],
                  'reg_lambda': hparams['reg_lambda'], 'learning_rate': hparams['learning_rate'],
                  'seed': hparams['seed'], 'eval_metric': 'rmse', 'verbosity': 1}

        # metrics
        metrics = [mt.Metric(mt.rms_score, np.nanmean),
                   mt.Metric(mt.concordance_index, np.nanmean),
                   mt.Metric(mt.pearson_r2_score, np.nanmean)]
        return params, {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}, metrics

    @staticmethod
    def data_provider(fold, flags, data):
        if not flags['cv']:
            print("Training scheme: train, validation" + (", test split" if flags['test'] else " split"))
            train_dataset = (data[1][0].X, data[1][0].y, data[1][0].w)
            valid_dataset = (data[1][1].X, data[1][1].y, data[1][1].w)
            test_dataset = None
            if flags['test']:
                test_dataset = (data[1][2].X, data[1][2].y, data[1][2].w)
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        else:
            train_dataset = (data[1][fold][0].X, data[1][fold][0].y, data[1][fold][0].w)
            valid_dataset = (data[1][fold][1].X, data[1][fold][1].y, data[1][fold][1].w)
            test_dataset = (data[1][fold][2].X, data[1][fold][2].y, data[1][fold][2].w)
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
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
    def train(xgb_params, data, metrics, transformer, simboost_feats_dict, tasks, n_iters=3000, is_hsearch=False,
              sim_data_node=None):
        start = time.time()
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        predicted_vals = []
        true_vals = []
        model_preds_node = DataNode(label="model_predictions", data={"y": true_vals,
                                                                     "y_pred": predicted_vals})
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node]

        # Data pre-processing
        xgb_data = {}
        for k in data:
            ds = data[k]
            dmat_x = []
            dmat_y = []
            for x, y, _ in zip(*ds):
                comp, prot = x
                dmat_x.append(simboost_feats_dict[Pair(comp, prot)])
                dmat_y.append(float(y))
            dmatrix = xgb.DMatrix(data=np.array(dmat_x), label=np.array(dmat_y))
            xgb_data[k] = dmatrix

        # training
        xgb_eval_results = {}
        eval_type = 'val' if is_hsearch else 'test'
        model = xgb.train(xgb_params, xgb_data['train'], n_iters,
                          [(xgb_data['train'], 'train'), (xgb_data[eval_type], eval_type)],
                          early_stopping_rounds=10, evals_result=xgb_eval_results)

        # evaluation
        y_hat = model.predict(xgb_data[eval_type]).reshape(-1, len(tasks))
        y_true = xgb_data[eval_type].get_label().reshape(-1, len(tasks))
        eval_dict = {}
        w = data[eval_type][2].reshape(-1, len(tasks))
        score = SimBoost.evaluate(eval_dict, y_true, y_hat, w, metrics, tasks, transformer)
        for m in eval_dict:
            if m in metrics_dict:
                metrics_dict[m].append(eval_dict[m])
            else:
                metrics_dict[m] = [eval_dict[m]]
        # apply transformers
        predicted_vals.extend(undo_transforms(y_hat, transformer).squeeze().tolist())
        true_vals.extend(undo_transforms(y_true, transformer).squeeze().tolist())
        scores_lst.append(score)
        print('Evaluation: score={}, metrics={}'.format(score, eval_dict))

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        return {'model': model, 'score': score, 'epoch': model.best_iteration}

    @staticmethod
    def evaluate_model(model_dir, model_name, data, metrics, transformer, simboost_feats_dict, tasks,
                       sim_data_node=None):
        start = time.time()
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        predicted_vals = []
        true_vals = []
        model_preds_node = DataNode(label="model_predictions", data={"y": true_vals,
                                                                     "y_pred": predicted_vals})
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node]

        # Data pre-processing
        xgb_data = {}
        for k in data:
            ds = data[k]
            dmat_x = []
            dmat_y = []
            for x, y, _ in zip(*ds):
                comp, prot = x
                dmat_x.append(simboost_feats_dict[Pair(comp, prot)])
                dmat_y.append(float(y))
            dmatrix = xgb.DMatrix(data=np.array(dmat_x), label=np.array(dmat_y))
            xgb_data[k] = dmatrix

        # load model
        model = load_xgboost(model_dir, model_name)

        # evaluation
        eval_type = 'test'
        y_hat = model.predict(xgb_data[eval_type]).reshape(-1, len(tasks))
        y_true = xgb_data[eval_type].get_label().reshape(-1, len(tasks))
        eval_dict = {}
        w = data[eval_type][2].reshape(-1, len(tasks))
        score = SimBoost.evaluate(eval_dict, y_true, y_hat, w, metrics, tasks, transformer)
        for m in eval_dict:
            if m in metrics_dict:
                metrics_dict[m].append(eval_dict[m])
            else:
                metrics_dict[m] = [eval_dict[m]]
        # apply transformers
        predicted_vals.extend(undo_transforms(y_hat, transformer).squeeze().tolist())
        true_vals.extend(undo_transforms(y_true, transformer).squeeze().tolist())
        scores_lst.append(score)
        print('Evaluation: score={}, metrics={}'.format(score, eval_dict))

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def save_xgboost(model, path, name):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(model, f)
    # with open(os.path.join(path, "simboost_dummy_save.txt"), 'a') as f:
    #     f.write(name + '\n')


def load_xgboost(path, name):
    return load_pickle(os.path.join(path, name))


def main(flags):
    if len(flags["views"]) > 0:
        print("Single views for training: {}, num={}".format(flags["views"], len(flags["views"])))
    else:
        print("No views selected for training")

    for view in flags["views"]:
        cview, pview = view
        sim_label = "SimBoost_{}_{}".format(cview, pview)
        print(sim_label)

        # Simulation data resource tree
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        dataset_lbl = flags["dataset_name"]
        node_label = "{}_{}_{}_{}_{}".format(dataset_lbl, sim_label, split_label, "eval" if flags["eval"] else "train",
                                             date_label)
        node_label = json.dumps({'model_family': 'simboost',
                                 'dataset': dataset_lbl,
                                 'cview': cview,
                                 'pview': pview,
                                 'split': split_label,
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'mode': "eval" if flags["eval"] else "train",
                                 'date': date_label})
        sim_data = DataNode(label=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])
        mf_simboost_data_dict = load_dict_model(flags['model_dir'], flags['mf_simboost_data_dict'])

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

            data_key = {"ecfp4": "SB_ECFP4",
                        "ecfp8": "SB_ECFP8"}.get(cview)
            data = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed,
                            mf_simboost_data_dict=mf_simboost_data_dict)
            tasks = data[0]
            flags["tasks"] = tasks
            simboost_feats_dict = data[5]
            transformer = data[2]
            drug_kernel_dict, prot_kernel_dict = data[4]

            trainer = SimBoost()

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
                                    "simboost_feats_dict": simboost_feats_dict,
                                    "tasks": tasks,
                                    "is_hsearch": True}

                hparams_conf = get_hparam_config(flags, seed)

                if hparam_search is None:
                    search_alg = {"random_search": RandomSearchCV,
                                  "bayopt_search": BayesianOptSearchCV}.get(flags["hparam_search_alg"],
                                                                            BayesianOptSearchCV)
                    min_opt = "gp"
                    hparam_search = search_alg(hparam_config=hparams_conf,
                                               num_folds=k,
                                               initializer=trainer.initialize,
                                               data_provider=trainer.data_provider,
                                               train_fn=trainer.train,
                                               save_model_fn=save_xgboost,
                                               init_args=extra_init_args,
                                               data_args=extra_data_args,
                                               train_args=extra_train_args,
                                               data_node=data_node,
                                               split_label=split_label,
                                               sim_label=sim_label,
                                               minimizer=min_opt,
                                               dataset_label=dataset_lbl,
                                               results_file="{}_{}_dti_{}_{}_{}.csv".format(
                                                   flags["hparam_search_alg"], sim_label, date_label, min_opt, n_iters))

                stats = hparam_search.fit(model_dir="models", model_name="".join(tasks), max_iter=20, seed=seed)
                print(stats)
                print("Best params = {}".format(stats.best()))
            else:
                invoke_train(trainer, tasks, data, transformer, flags, data_node, sim_label, simboost_feats_dict, seed)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data, transformer, flags, data_node, view, simboost_feats_dict, seed):
    hyper_params = default_hparams_bopt(flags, seed)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data, flags, hyper_params, tasks, trainer, transformer, view, simboost_feats_dict, k)
    else:
        start_fold(data_node, data, flags, hyper_params, tasks, trainer, transformer, view, simboost_feats_dict)


def start_fold(sim_data_node, data, flags, hyper_params, tasks, trainer, transformer, view, simboost_feats_dict,
               k=None):
    data = trainer.data_provider(k, flags, data)
    model, data, metrics = trainer.initialize(hparams=hyper_params, train_dataset=data["train"],
                                              val_dataset=data["val"], test_dataset=data["test"])
    if flags["eval"]:
        trainer.evaluate_model(flags.model_dir, flags.eval_model_name, data, metrics, transformer, simboost_feats_dict,
                               tasks, sim_data_node)
    else:
        # Train the model
        results = trainer.train(model, data, metrics, transformer, simboost_feats_dict, tasks=tasks,
                                sim_data_node=sim_data_node, n_iters=10000)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = "warm" if flags["split_warm"] else "cold_target" if flags["cold_target"] else "cold_drug" if \
            flags["cold_drug"] else "None"
        save_xgboost(model, flags["model_dir"],
                     "{}_{}_{}_{}_{}_{:.4f}".format(flags["dataset_name"], view, flags["model_name"], split_label,
                                                    epoch,
                                                    score))


def default_hparams_rand(flags, seed):
    return {
        "reg_lambda": 0.1,
    }


def default_hparams_bopt(flags, seed):
    return {
        'seed': seed,
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'subsample': 1.0,
        'colsample_bytree': .5,
        'n_estimators': 50,
        'gamma': .1,
        'reg_lambda': 0.7427012361969033,
        'learning_rate': 0.1078028400493326,
    }


def get_hparam_config(flags, seed):
    return {
        'seed': ConstantParam(seed),
        'objective': ConstantParam('reg:squarederror'),  # reg:linear
        'max_depth': DiscreteParam(5, 10),
        'subsample': RealParam(min=0.5),
        'colsample_bytree': RealParam(min=0.5),
        'n_estimators': DiscreteParam(min=50, max=200),
        'gamma': RealParam(min=0.1),
        'reg_lambda': RealParam(min=0.1),
        'learning_rate': LogRealParam(),
    }


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
    parser.add_argument("--mf_simboost_data_dict",
                        type=str,
                        help="The filename of the Matrix Factorization drug-target features dict to be "
                             "loaded from the directory specified in --model_dir")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    setattr(flags, "cv", True if flags.fold_num > 2 else False)
    setattr(flags, "views", [(cv, pv) for cv, pv in zip(args.comp_view, args.prot_view)])
    split = 'warm'
    flags['split'] = split
    flags['predict_cold'] = split == 'cold_drug_target'
    flags['cold_drug'] = split == 'cold_drug'
    flags['cold_target'] = split == 'cold_target'
    flags['cold_drug_cluster'] = split == 'cold_drug_cluster'
    flags['split_warm'] = split == 'warm'
    main(flags)
