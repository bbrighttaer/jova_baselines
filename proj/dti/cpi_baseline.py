# Author: bbrighttaer
# Project: jova
# Date: 10/28/19
# Time: 10:13am
# File: cpi_baseline.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import json
import random
import time
from datetime import datetime as dt
from itertools import chain

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from soek import *
from soek.bopt import GPMinArgs
from torch.utils.data import DataLoader
from tqdm import tqdm

import jova.metrics as mt
import jova.utils.io
from jova import cuda
from jova.data import batch_collator, get_data, load_proteins, DtiDataset
from jova.metrics import compute_model_performance
from jova.nn.layers import GraphConvLayer, GraphPool, GraphGather
from jova.nn.models import create_fcn_layers, WeaveModel, GraphConvSequential, ProteinCNNAttention, \
    ProtCnnForward, GraphNeuralNet, Prot2Vec
from jova.trans import undo_transforms
from jova.utils import Trainer
from jova.utils.args import FcnArgs, WeaveLayerArgs, WeaveGatherArgs
from jova.utils.io import save_model, load_model, load_pickle
from jova.utils.math import ExpAverage
from jova.utils.train_helpers import count_parameters, FrozenModels, np_to_plot_data

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]  # , 8, 64]

if cuda:
    torch.cuda.set_device(0)

def create_ecfp_net(hparams):
    fcn_args = []
    for i in range(len(hparams["ecfp"]["ecfp_hdims"]) - 1):
        conf = FcnArgs(in_features=hparams["ecfp"]["ecfp_hdims"][i],
                       out_features=hparams["ecfp"]["ecfp_hdims"][i + 1],
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"])
        fcn_args.append(conf)
    fcn_args.append(FcnArgs(in_features=hparams["ecfp"]["ecfp_hdims"][-1], out_features=hparams["latent_dim"]))
    fcn_args.append(FcnArgs(in_features=hparams["latent_dim"], out_features=1))
    model = nn.Sequential(*create_fcn_layers(fcn_args))
    return model


def create_weave_net(hparams):
    weave_args = (
        WeaveLayerArgs(n_atom_input_feat=75,
                       n_pair_input_feat=14,
                       n_atom_output_feat=hparams["weave"]["dim"],
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=hparams["weave"]["update_pairs"],
                       activation='relu',
                       batch_norm=True,
                       dropout=hparams["dprob"]
                       ),
        WeaveLayerArgs(n_atom_input_feat=50,
                       n_pair_input_feat=14,
                       n_atom_output_feat=hparams["weave"]["dim"],
                       n_pair_output_feat=50,
                       n_hidden_AA=50,
                       n_hidden_PA=50,
                       n_hidden_AP=50,
                       n_hidden_PP=50,
                       update_pair=hparams["weave"]["update_pairs"],
                       batch_norm=True,
                       dropout=hparams["dprob"],
                       activation='relu'),
    )
    wg_args = WeaveGatherArgs(conv_out_depth=hparams["weave"]["dim"],
                              gaussian_expand=True, n_depth=hparams["latent_dim"])
    weave_model = WeaveModel(weave_args, weave_gath_arg=wg_args, weave_type='1D')
    return weave_model


def create_gconv_net(hparams):
    dim = hparams["gconv"]["dim"]
    gconv_model = GraphConvSequential(GraphConvLayer(in_dim=75, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      GraphConvLayer(in_dim=64, out_dim=64),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      GraphPool(),

                                      nn.Linear(in_features=64, out_features=dim),
                                      nn.BatchNorm1d(dim),
                                      nn.ReLU(),
                                      nn.Dropout(hparams["dprob"]),
                                      GraphGather(activation='tanh'))
    return nn.Sequential(gconv_model, nn.Linear(dim * 2, hparams["latent_dim"]),
                         nn.BatchNorm1d(hparams["latent_dim"]), nn.ReLU())


def create_gnn_net(hparams):
    dim = hparams["gnn"]["dim"]
    gnn_model = GraphNeuralNet(num_fingerprints=hparams["gnn"]["fingerprint_size"], embedding_dim=dim,
                               num_layers=hparams["gnn"]["num_layers"])
    return nn.Sequential(gnn_model, nn.Linear(dim, hparams["prot"]["dim"]),
                         nn.BatchNorm1d(hparams["prot"]["dim"]), nn.ReLU(), nn.Dropout(hparams["dprob"]))


class CPIBaseline(Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, test_dataset, protein_profile, cuda_devices=None,
                   mode="regression"):
        frozen_models = FrozenModels()

        # create network
        view_lbl = hparams["view"]
        create_comp_model = {"ecfp4": create_ecfp_net,
                             "ecfp8": create_ecfp_net,
                             "weave": create_weave_net,
                             "gconv": create_gconv_net,
                             "gnn": create_gnn_net}.get(view_lbl)

        comp_model = create_comp_model(hparams)
        # pt_embeddings = create_torch_embeddings(frozen_models_hook=frozen_models,
        #                                         np_embeddings=protein_embeddings)
        func_callback = None
        comp_net_pcnn = ProtCnnForward(prot2vec=Prot2Vec(protein_profile=protein_profile,
                                                         vocab_size=hparams["prot"]["vocab_size"],
                                                         embedding_dim=hparams["prot"]["dim"],
                                                         batch_first=True),
                                       prot_cnn_model=ProteinCNNAttention(dim=hparams["prot"]["dim"],
                                                                          window=hparams["prot"]["window"],
                                                                          num_layers=hparams["prot"][
                                                                              "prot_cnn_num_layers"],
                                                                          attn_hook=func_callback),
                                       comp_model=comp_model)

        p = 2 * hparams["prot"]["dim"]
        layers = [comp_net_pcnn]
        for dim in hparams["hdims"]:
            layers.append(nn.Linear(p, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hparams["dprob"]))
            p = dim

        # Output layer
        layers.append(nn.Linear(in_features=p, out_features=hparams["output_dim"]))

        model = nn.Sequential(*layers)

        print("Number of trainable parameters = {}".format(count_parameters(model)))
        if cuda:
            model = model.cuda()

        # data loaders
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=10 if hparams["explain_mode"] else hparams["tr_batch_size"],
                                       shuffle=True,
                                       collate_fn=lambda x: x)
        val_data_loader = DataLoader(dataset=val_dataset,
                                     batch_size=10 if hparams["explain_mode"] else hparams["val_batch_size"],
                                     shuffle=False,
                                     collate_fn=lambda x: x)
        test_data_loader = None
        if test_dataset is not None:
            test_data_loader = DataLoader(dataset=test_dataset,
                                          batch_size=10 if hparams["explain_mode"] else hparams["test_batch_size"],
                                          shuffle=False,
                                          collate_fn=lambda x: x)

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
        return model, optimizer, {"train": train_data_loader,
                                  "val": val_data_loader,
                                  "test": test_data_loader}, metrics, frozen_models

    @staticmethod
    def data_provider(fold, flags, data_dict):
        if not flags['cv']:
            print("Training scheme: train, validation" + (", test split" if flags['test'] else " split"))
            train_dataset = DtiDataset(x_s=[data[1][0].X for data in data_dict.values()],
                                       y_s=[data[1][0].y for data in data_dict.values()],
                                       w_s=[data[1][0].w for data in data_dict.values()])
            valid_dataset = DtiDataset(x_s=[data[1][1].X for data in data_dict.values()],
                                       y_s=[data[1][1].y for data in data_dict.values()],
                                       w_s=[data[1][1].w for data in data_dict.values()])
            test_dataset = None
            if flags['test']:
                test_dataset = DtiDataset(x_s=[data[1][2].X for data in data_dict.values()],
                                          y_s=[data[1][2].y for data in data_dict.values()],
                                          w_s=[data[1][2].w for data in data_dict.values()])
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        else:
            train_dataset = DtiDataset(x_s=[data[1][fold][0].X for data in data_dict.values()],
                                       y_s=[data[1][fold][0].y for data in data_dict.values()],
                                       w_s=[data[1][fold][0].w for data in data_dict.values()])
            valid_dataset = DtiDataset(x_s=[data[1][fold][1].X for data in data_dict.values()],
                                       y_s=[data[1][fold][1].y for data in data_dict.values()],
                                       w_s=[data[1][fold][1].w for data in data_dict.values()])
            test_dataset = DtiDataset(x_s=[data[1][fold][2].X for data in data_dict.values()],
                                      y_s=[data[1][fold][2].y for data in data_dict.values()],
                                      w_s=[data[1][fold][2].w for data in data_dict.values()])
            data = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
        return data

    @staticmethod
    def evaluate(eval_dict, y, y_pred, w, metrics, tasks, transformers):
        eval_dict.update(compute_model_performance(metrics, y_pred.cpu().detach().numpy(), y, w, transformers,
                                                   tasks=tasks))
        # scoring
        rms = np.nanmean(eval_dict["nanmean-rms_score"])
        ci = np.nanmean(eval_dict["nanmean-concordance_index"])
        r2 = np.nanmean(eval_dict["nanmean-pearson_r2_score"])
        score = np.nanmean([ci, r2]) - rms
        return score

    @staticmethod
    def train(model, optimizer, data_loaders, metrics, frozen_models, transformers_dict, prot_desc_dict, tasks,
              view_lbl, n_iters=5000, sim_data_node=None, epoch_ckpt=(2, 1.0), tb_writer=None, is_hsearch=False):
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        terminate_training = False
        e_avg = ExpAverage(.01)
        n_epochs = n_iters // len(data_loaders["train"])
        scheduler = sch.StepLR(optimizer, step_size=400, gamma=0.01)
        criterion = torch.nn.MSELoss()

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node]
        try:
            # Main training loop
            for epoch in range(0):
                if terminate_training:
                    print("Terminating training...")
                    break
                for phase in ["train", "val" if is_hsearch else "test"]:
                    # ensure these models are frozen at all times
                    for m in frozen_models:
                        m.eval()

                    if phase == "train":
                        print("Training....")
                        # Training mode
                        model.train()
                    else:
                        print("Validation...")
                        # Evaluation mode
                        model.eval()

                    data_size = 0.
                    epoch_losses = []
                    epoch_scores = []

                    # Iterate through mini-batches
                    i = 0
                    for batch in tqdm(data_loaders[phase]):
                        batch_size, data = batch_collator(batch, prot_desc_dict, spec=view_lbl,
                                                          cuda_prot=False)
                        # Data
                        protein_x = data[view_lbl][0][2]
                        if view_lbl == "gconv":
                            # graph data structure is: [(compound data, batch_size), protein_data]
                            X = ((data[view_lbl][0][0], batch_size), protein_x)
                        else:
                            X = (data[view_lbl][0][0], protein_x)
                        y = np.array([k for k in data[view_lbl][1]], dtype=np.float)
                        w = np.array([k for k in data[view_lbl][2]], dtype=np.float)

                        optimizer.zero_grad()

                        # forward propagation
                        # track history if only in train
                        with torch.set_grad_enabled(phase == "train"):
                            outputs = model(X)
                            target = torch.from_numpy(y).float()
                            weights = torch.from_numpy(w).float()
                            if cuda:
                                target = target.cuda()
                                weights = weights.cuda()
                            outputs = outputs * weights
                            loss = criterion(outputs, target)

                        if phase == "train":
                            print("\tEpoch={}/{}, batch={}/{}, loss={:.4f}".format(epoch + 1, n_epochs, i + 1,
                                                                                   len(data_loaders[phase]),
                                                                                   loss.item()))
                            # for epoch stats
                            epoch_losses.append(loss.item())

                            # for sim data resource
                            loss_lst.append(loss.item())

                            # optimization ops
                            loss.backward()
                            optimizer.step()
                        else:
                            if str(loss.item()) != "nan":  # useful in hyperparameter search
                                eval_dict = {}
                                score = CPIBaseline.evaluate(eval_dict, y, outputs, w, metrics, tasks,
                                                             transformers_dict[view_lbl])
                                # for epoch stats
                                epoch_scores.append(score)

                                # for sim data resource
                                scores_lst.append(score)
                                for m in eval_dict:
                                    if m in metrics_dict:
                                        metrics_dict[m].append(eval_dict[m])
                                    else:
                                        metrics_dict[m] = [eval_dict[m]]

                                print("\nEpoch={}/{}, batch={}/{}, "
                                      "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                                len(data_loaders[phase]),
                                                                                eval_dict, score))
                            else:
                                terminate_training = True

                        i += 1
                        data_size += batch_size
                    # End of mini=batch iterations.

                    if phase == "train":
                        ep_loss = np.nanmean(epoch_losses)
                        e_avg.update(ep_loss)
                        if epoch % (epoch_ckpt[0] - 1) == 0 and epoch > 0:
                            if e_avg.value > epoch_ckpt[1]:
                                terminate_training = True
                        # Adjust the learning rate.
                        scheduler.step()
                        print("\nPhase: {}, avg task loss={:.4f}, ".format(phase, np.nanmean(epoch_losses)))
                    else:
                        mean_score = np.mean(epoch_scores)
                        if best_score < mean_score:
                            best_score = mean_score
                            best_model_wts = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
        except RuntimeError as e:
            print(str(e))
        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        model.load_state_dict(best_model_wts)
        return {'model': model, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(model, model_dir, model_name, data_loaders, metrics, transformers_dict, prot_desc_dict,
                       tasks, view_lbl, sim_data_node=None):
        # load saved model and put in evaluation mode
        model.load_state_dict(load_model(model_dir, model_name))
        model.eval()

        print("Model evaluation...")
        start = time.time()
        n_epochs = 1

        # sub-nodes of sim data resource
        # loss_lst = []
        # train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)
        predicted_vals = []
        true_vals = []
        model_preds_node = DataNode(label="model_predictions", data={"y": true_vals,
                                                                     "y_pred": predicted_vals})

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [metrics_node, scores_node, model_preds_node]

        # Main evaluation loop
        for epoch in range(n_epochs):

            for phase in ["test"]:
                # Iterate through mini-batches
                i = 0
                for batch in tqdm(data_loaders[phase]):
                    batch_size, data = batch_collator(batch, prot_desc_dict, spec=view_lbl,
                                                      cuda_prot=False)
                    # Data
                    protein_x = data[view_lbl][0][2]
                    if view_lbl == "gconv":
                        # graph data structure is: [(compound data, batch_size), protein_data]
                        X = ((data[view_lbl][0][0], batch_size), protein_x)
                    else:
                        X = (data[view_lbl][0][0], protein_x)
                    y_true = np.array([k for k in data[view_lbl][1]], dtype=np.float)
                    w = np.array([k for k in data[view_lbl][2]], dtype=np.float)

                    # forward propagation
                    with torch.set_grad_enabled(False):
                        y_predicted = model(X)

                        # apply transformers
                        predicted_vals.extend(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                              transformers_dict[view_lbl]).squeeze().tolist())
                        true_vals.extend(undo_transforms(y_true,
                                                         transformers_dict[view_lbl]).astype(
                            np.float).squeeze().tolist())

                    eval_dict = {}
                    score = CPIBaseline.evaluate(eval_dict, y_true, y_predicted, w, metrics, tasks,
                                                 transformers_dict[view_lbl])

                    # for sim data resource
                    scores_lst.append(score)
                    for m in eval_dict:
                        if m in metrics_dict:
                            metrics_dict[m].append(eval_dict[m])
                        else:
                            metrics_dict[m] = [eval_dict[m]]

                    print("\nEpoch={}/{}, batch={}/{}, "
                          "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                    len(data_loaders[phase]),
                                                                    eval_dict, score))

                    i += 1
                # End of mini=batch iterations.

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    @staticmethod
    def explain_model(model, model_dir, model_name, data_loaders, transformers_dict, prot_desc_dict, view_lbl,
                      sim_data_node=None, max_print=10000, k=10):
        # load saved model and put in evaluation mode
        model.load_state_dict(jova.utils.io.load_model(model_dir, model_name, dvc='cuda' if torch.cuda.is_available()
        else 'cpu'))
        model.eval()

        print("Model evaluation...")
        start = time.time()

        # sub-nodes of sim data resource
        attn_ranking = []
        attn_ranking_node = DataNode(label="attn_ranking", data=attn_ranking)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [attn_ranking_node]

        # Iterate through mini-batches
        i = 0
        # Since we're evaluating, join all data loaders
        all_loaders = chain()
        for loader in data_loaders:
            if data_loaders[loader] is not None:
                all_loaders = chain(all_loaders, data_loaders[loader])

        for batch in tqdm(all_loaders):
            if i == max_print:
                print('\nMaximum number [%d] of samples limit reached. Terminating...' % i)
                break
            i += 1
            batch_size, data = batch_collator(batch, prot_desc_dict, spec=view_lbl,
                                              cuda_prot=False)

            # Data
            protein_x = data[view_lbl][0][2]
            if view_lbl == "gconv":
                # graph data structure is: [(compound data, batch_size), protein_data]
                X = ((data[view_lbl][0][0], batch_size), protein_x)
            else:
                X = (data[view_lbl][0][0], protein_x)
            y_true = np.array([k for k in data[view_lbl][1]], dtype=np.float)
            w = np.array([k for k in data[view_lbl][2]], dtype=np.float)

            # attention x data for analysis
            attn_data_x = {}
            attn_data_x['pcnna'] = protein_x

            # forward propagation
            with torch.set_grad_enabled(False):
                y_predicted = model(X)

            # get segments ranking
            transformer = transformers_dict[view_lbl]
            rank_results = {'y_pred': np_to_plot_data(undo_transforms(y_predicted.cpu().detach().numpy(),
                                                                      transformer)),
                            'y_true': np_to_plot_data(undo_transforms(y_true, transformer)),
                            'smiles': data[view_lbl][0][3][0][0].smiles}
            attn_ranking.append(rank_results)
        # End of mini=batch iterations.

        duration = time.time() - start
        print('\nPrediction interpretation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))


def main(pid, flags):
    if len(flags.views) > 0:
        print("Single views for training:", flags.views)
    else:
        print("No views selected for training")

    for view in flags.views:
        sim_label = "cpi_prediction_baseline"
        print("CUDA={}, view={}".format(cuda, view))

        # Simulation data resource tree
        split_label = flags.split
        dataset_lbl = flags["dataset_name"]
        if flags['eval']:
            mode = 'eval'
        elif flags['explain']:
            mode = 'explain'
        else:
            mode = 'train'
        node_label = json.dumps({'model_family': 'cpi',
                                 'dataset': dataset_lbl,
                                 'cview': 'gnn',
                                 'pview': 'pcnna',
                                 'split': split_label,
                                 'cv': flags["cv"],
                                 'seeds': '-'.join([str(s) for s in seeds]),
                                 'mode': mode,
                                 'date': date_label})
        sim_data = DataNode(label='_'.join([sim_label, dataset_lbl, split_label, mode,
                                            date_label]), metadata=node_label)
        nodes_list = []
        sim_data.data = nodes_list

        num_cuda_dvcs = torch.cuda.device_count()
        cuda_devices = None if num_cuda_dvcs == 1 else [i for i in range(1, num_cuda_dvcs)]

        # Runtime Protein stuff
        prot_desc_dict, prot_seq_dict = load_proteins(flags['prot_desc_path'])
        prot_profile, prot_vocab = load_pickle(file_name=flags.prot_profile), load_pickle(file_name=flags.prot_vocab)
        flags["prot_vocab_size"] = len(prot_vocab)

        # For searching over multiple seeds
        hparam_search = None

        for seed in seeds:
            # for data collection of this round of simulation.
            data_node = DataNode(label="seed_%d" % seed)
            nodes_list.append(data_node)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # load data
            print('-------------------------------------')
            print('Running on dataset: %s' % dataset_lbl)
            print('-------------------------------------')

            data_dict = dict()
            transformers_dict = dict()
            data_key = {"ecfp4": "ECFP4",
                        "ecfp8": "ECFP8",
                        "weave": "Weave",
                        "gconv": "GraphConv",
                        "gnn": "GNN"}.get(view)
            data_dict[view] = get_data(data_key, flags, prot_sequences=prot_seq_dict, seed=seed)
            transformers_dict[view] = data_dict[view][2]
            flags["gnn_fingerprint"] = data_dict[view][3]

            tasks = data_dict[view][0]
            flags["tasks"] = tasks

            trainer = CPIBaseline()

            if flags["cv"]:
                k = flags["fold_num"]
                print("{}, {}-Prot: Training scheme: {}-fold cross-validation".format(tasks, view, k))
            else:
                k = 1
                print("{}, {}-Prot: Training scheme: train, validation".format(tasks, view)
                      + (", test split" if flags['test'] else " split"))

            if flags["hparam_search"]:
                print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

                # arguments to callables
                extra_init_args = {"mode": "regression",
                                   "cuda_devices": cuda_devices,
                                   "protein_profile": prot_profile}
                extra_data_args = {"flags": flags,
                                   "data_dict": data_dict}
                extra_train_args = {"transformers_dict": transformers_dict,
                                    "prot_desc_dict": prot_desc_dict,
                                    "tasks": tasks,
                                    "n_iters": 3000,
                                    "is_hsearch": True,
                                    "view_lbl": view}

                hparams_conf = get_hparam_config(flags, view)
                if hparam_search is None:
                    search_alg = {"random_search": RandomSearch,
                                  "bayopt_search": BayesianOptSearch}.get(flags["hparam_search_alg"],
                                                                          BayesianOptSearch)
                    search_args = GPMinArgs(n_calls=20)
                    min_opt = "gbrt"
                    hparam_search = search_alg(hparam_config=hparams_conf,
                                               num_folds=k,
                                               initializer=trainer.initialize,
                                               data_provider=trainer.data_provider,
                                               train_fn=trainer.train,
                                               save_model_fn=jova.utils.io.save_model,
                                               init_args=extra_init_args,
                                               data_args=extra_data_args,
                                               train_args=extra_train_args,
                                               alg_args=search_args,
                                               data_node=data_node,
                                               split_label=split_label,
                                               sim_label=sim_label,
                                               minimizer=min_opt,
                                               dataset_label=dataset_lbl,
                                               results_file="{}_{}_dti_{}_{}.csv".format(
                                                   flags["hparam_search_alg"], sim_label, date_label, min_opt))

                stats = hparam_search.fit()
                print(stats)
                print("Best params = {}".format(stats.best()))
            else:
                invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node,
                             view, prot_profile)

        # save simulation data resource tree to file.
        sim_data.to_json(path="./analysis/")


def invoke_train(trainer, tasks, data_dict, transformers_dict, flags, prot_desc_dict, data_node, view,
                 prot_profile):
    hyper_params = default_hparams_bopt(flags, view)
    # Initialize the model and other related entities for training.
    if flags["cv"]:
        folds_data = []
        data_node.data = folds_data
        data_node.label = data_node.label + "cv"
        for k in range(flags["fold_num"]):
            k_node = DataNode(label="fold-%d" % k)
            folds_data.append(k_node)
            start_fold(k_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                       transformers_dict, view, prot_profile, k)
    else:
        start_fold(data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
                   transformers_dict, view, prot_profile)


def start_fold(sim_data_node, data_dict, flags, hyper_params, prot_desc_dict, tasks, trainer,
               transformers_dict, view, prot_profile, k=None):
    data = trainer.data_provider(k, flags, data_dict)
    model, optimizer, data_loaders, metrics, frozen_models = trainer.initialize(hparams=hyper_params,
                                                                                train_dataset=data["train"],
                                                                                val_dataset=data["val"],
                                                                                test_dataset=data["test"],
                                                                                protein_profile=prot_profile)
    if flags["eval"]:
        trainer.evaluate_model(model, flags["model_dir"], flags["eval_model_name"],
                               data_loaders, metrics, transformers_dict,
                               prot_desc_dict, tasks, view_lbl=view, sim_data_node=sim_data_node)
    elif flags["explain"]:
        trainer.explain_model(model, flags["model_dir"], flags["eval_model_name"], data_loaders, transformers_dict,
                              prot_desc_dict, view, sim_data_node)
    else:
        # Train the model
        results = trainer.train(model, optimizer, data_loaders, metrics, frozen_models,
                                transformers_dict, prot_desc_dict, tasks, n_iters=10000, view_lbl=view,
                                sim_data_node=sim_data_node)
        model, score, epoch = results['model'], results['score'], results['epoch']
        # Save the model.
        split_label = flags.split
        save_model(model, flags["model_dir"],
                   "{}_pcnna_{}_{}_{}_{}_{:.4f}".format(flags["dataset_name"], view, flags["model_name"], split_label,
                                                        epoch, score))


def default_hparams_rand(flags, view):
    return {
        "view": view,
        "prot_dim": 8421,
        "comp_dim": 1024,
        "hdims": [3795, 2248, 2769, 2117],

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.0739227,

        "tr_batch_size": 256,
        "val_batch_size": 512,
        "test_batch_size": 512,

        # optimizer params
        "optimizer": "rmsprop",
        "optimizer__sgd__weight_decay": 1e-4,
        "optimizer__sgd__nesterov": True,
        "optimizer__sgd__momentum": 0.9,
        "optimizer__sgd__lr": 1e-3,

        "optimizer__adam__weight_decay": 1e-4,
        "optimizer__adam__lr": 1e-3,

        "optimizer__rmsprop__lr": 0.000235395,
        "optimizer__rmsprop__weight_decay": 0.000146688,
        "optimizer__rmsprop__momentum": 0.00622082,
        "optimizer__rmsprop__centered": False
    }


def default_hparams_bopt(flags, view):
    return {
        "explain_mode": flags.explain,
        "view": view,
        "hdims": [2092],
        "output_dim": len(flags.tasks),

        # weight initialization
        "kaiming_constant": 5,

        # dropout regs
        "dprob": 0.12967965527359,

        "tr_batch_size": 256,
        "val_batch_size": 128,
        "test_batch_size": 128,

        # optimizer params
        "optimizer": "adamax",
        "optimizer__global__weight_decay": 0.0016123580093276922,
        "optimizer__global__lr": 0.00330365625227763,
        "optimizer__adadelta__rho": 0.115873,

        "prot": {
            "dim": 30,
            "vocab_size": flags["prot_vocab_size"],
            "window": 11,
            "prot_cnn_num_layers": 3
        },
        "weave": {
            "dim": 50,
            "update_pairs": False,
        },
        "gconv": {
            "dim": 512,
        },
        "ecfp8": {
            "in_dim": 1024,
            "ecfp_hdims": [653, 3635],
        },
        "gnn": {
            "fingerprint_size": len(flags["gnn_fingerprint"]) if flags['gnn_fingerprint_size'] is None else
            flags['gnn_fingerprint_size'],
            "num_layers": 2,
            "dim": 500,
        }
    }


def get_hparam_config(flags, view):
    return {
        "explain_mode": ConstantParam(flags.explain),
        "view": ConstantParam(view),
        "hdims": DiscreteParam(min=256, max=5000, size=DiscreteParam(min=1, max=4)),
        "output_dim": ConstantParam(len(flags.tasks)),

        # weight initialization
        "kaiming_constant": ConstantParam(5),

        # dropout regs
        "dprob": RealParam(min=0.1),

        "tr_batch_size": CategoricalParam([1, 32, 64, 128, 256]),
        "val_batch_size": ConstantParam(128),
        "test_batch_size": ConstantParam(128),

        # optimizer params
        "optimizer": CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": LogRealParam(),
        "optimizer__global__lr": LogRealParam(),

        "prot": DictParam({
            "dim": DiscreteParam(min=5, max=50),
            "vocab_size": ConstantParam(flags["prot_vocab_size"]),
            "window": ConstantParam(11),
            "prot_cnn_num_layers": DiscreteParam(min=1, max=4)
        }),
        "weave": ConstantParam({
            "dim": 50,
            "update_pairs": False,
        }),
        "gconv": ConstantParam({
            "dim": 512,
        }),
        "ecfp8": ConstantParam({
            "in_dim": 1024,
            "ecfp_hdims": [653, 3635],
        }),
        "gnn": DictParam({
            "fingerprint_size": ConstantParam(len(flags["gnn_fingerprint"])),
            "num_layers": DiscreteParam(1, 4),
            "dim": DiscreteParam(min=64, max=512),
        })
    }


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DTI with jova model training.")

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
    parser.add_argument('--prot_profile',
                        type=str,
                        help='A resource for retrieving embedding indexing profile of proteins.'
                        )
    parser.add_argument('--prot_vocab',
                        type=str,
                        help='A resource containing all N-gram segments/words constructed from the protein sequences.'
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
    parser.add_argument("--view",
                        action="append",
                        dest="views",
                        help="The view to be simulated. One of [ecfp4, ecfp8, weave, gconv]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated using CV")
    parser.add_argument("--explain",
                        action="store_true",
                        help="If true, a saved model is loaded and used in ranking segments to explain predictions")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")
    parser.add_argument('--gnn_fingerprint_size', default=None, type=int,
                        help='The size of the fingerprint dictionary (if a gnn fingerprint size other than that of'
                             ' the dataset is to be used)')
    # parser.add_argument("--fingerprint",
    #                     default=None,
    #                     type=str,
    #                     help="The pickled python dictionary containing the GNN fingerprint profiles of atoms and their"
    #                          "neighbors")
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
        flags['split'] = split
        flags['predict_cold'] = split == 'cold_drug_target'
        flags['cold_drug'] = split == 'cold_drug'
        flags['cold_target'] = split == 'cold_target'
        flags['cold_drug_cluster'] = split == 'cold_drug_cluster'
        flags['split_warm'] = split == 'warm'
        if use_mp:
            p = mp.spawn(fn=main, args=(flags,), join=False)
            procs.append(p)
            # p.start()
        else:
            main(0, flags)
    for proc in procs:
        proc.join()
