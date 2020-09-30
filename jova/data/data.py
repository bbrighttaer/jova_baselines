# Author: bbrighttaer
# Project: jova
# Date: 5/20/19
# Time: 1:17 PM
# File: data.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import time
from collections import defaultdict
from math import sqrt

import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio import Align
from rdkit.Chem import DataStructs
from torch.utils.data import dataset as ds
from tqdm import tqdm

from jova import cuda as _cuda
from jova.data import load_csv_dataset
from jova.feat.mol_graphs import ConvMol
from jova.utils.math import block_diag_irregular
from jova.utils.thread import UnboundedProgressbar
from jova.utils.train_helpers import ViewsReg


def load_prot_dict(prot_desc_dict, prot_seq_dict, prot_desc_path,
                   sequence_field, phospho_field):
    if re.search('davis', prot_desc_path, re.I):
        source = 'davis'
    elif re.search('metz', prot_desc_path, re.I):
        source = 'metz'
    elif re.search('kiba', prot_desc_path, re.I):
        source = 'kiba'
    elif re.search('toxcast', prot_desc_path, re.I):
        source = 'toxcast'
    elif re.search('human', prot_desc_path, re.I):
        source = 'human'
    elif re.search('celegans', prot_desc_path, re.I):
        source = 'celegans'
    elif re.search('egfr_case_study', prot_desc_path, re.I) or re.search('egfr_unfiltered', prot_desc_path, re.I) \
            or re.search('egfr_1M17', prot_desc_path, re.I):
        source = 'egfr_cs'

    df = pd.read_csv(prot_desc_path, index_col=0)
    # protList = list(df.index)
    for row in df.itertuples():
        descriptor = row[2:]
        descriptor = np.array(descriptor)
        descriptor = np.reshape(descriptor, (1, len(descriptor)))
        pair = (source, row[0])
        assert pair not in prot_desc_dict
        prot_desc_dict[pair] = descriptor
        sequence = row[sequence_field]
        phosphorylated = row[phospho_field]
        assert pair not in prot_seq_dict
        prot_seq_dict[pair] = (phosphorylated, sequence)


def load_dti_data(featurizer, dataset_name, dataset_file, prot_seq_dict, input_protein=True, cross_validation=False,
                  test=False, fold_num=5, split='random', reload=True, predict_cold=False, cold_drug=False,
                  cold_target=False, cold_drug_cluster=False, split_warm=False, filter_threshold=0,
                  mode='regression', seed=0, mf_simboost_data_dict=None):
    if cross_validation:
        test = False
    tasks, all_dataset, transformers, \
    fp, kernel_dicts, simboost_feats, MF_entities_dict = load_csv_dataset(dataset_name, dataset_file,
                                                                          featurizer=featurizer,
                                                                          cross_validation=cross_validation,
                                                                          test=test, split=split,
                                                                          reload=reload,
                                                                          K=fold_num, mode=mode,
                                                                          predict_cold=predict_cold,
                                                                          cold_drug=cold_drug,
                                                                          cold_target=cold_target,
                                                                          cold_drug_cluster=cold_drug_cluster,
                                                                          split_warm=split_warm,
                                                                          prot_seq_dict=prot_seq_dict,
                                                                          filter_threshold=filter_threshold,
                                                                          input_protein=input_protein,
                                                                          seed=seed,
                                                                          mf_simboost_data_dict=mf_simboost_data_dict)
    return tasks, all_dataset, transformers, fp, kernel_dicts, simboost_feats, MF_entities_dict


def load_proteins(prot_desc_path):
    """
    Retrieves all proteins in the tuple of paths given.

    :param prot_desc_path: A tuple of file paths containing the protein (PSC) descriptors.
    :return: A set of dicts: (descriptor information, sequence information)
    """
    prot_desc_dict = {}
    prot_seq_dict = {}
    for path in prot_desc_path:
        load_prot_dict(prot_desc_dict, prot_seq_dict, path, 1, 2)
    return prot_desc_dict, prot_seq_dict


class DtiDataset(ds.Dataset):

    def __init__(self, x_s, y_s, w_s):
        """
        Creates a Drug-Target Indication dataset object.

        :param x_s: a tuple of X data of each view.
        :param y_s: a tuple of y data of each view.
        :param w_s: a tuple of label weights of each view.
        """
        assert len(x_s) == len(y_s) == len(w_s), "Number of views in x_s must be equal to that of y_s."
        self.x_s = x_s
        self.y_s = y_s
        self.w_s = w_s

    def __len__(self):
        return len(self.x_s[0])

    def __getitem__(self, index):
        x_s = []
        y_s = []
        w_s = []
        for view_x, view_y, view_w in zip(self.x_s, self.y_s, self.w_s):
            x_s.append(view_x[index])
            y_s.append(view_y[index])
            w_s.append(view_w[index])
        return x_s, y_s, w_s


class Dataset(ds.Dataset):
    """Wrapper for the dataset to pytorch models"""

    def __init__(self, views_data):
        """
        Creates a dataset wrapper.

        :param views_data: Data of all views. Structure: ((X1, Y1), (X2, Y2), ...)
        """
        self.X_list = []
        self.y_list = []
        self.num_views = len(views_data)
        for data in views_data:
            self.X_list.append(data[0])  # 0 -> X data
            self.y_list.append(data[1])  # 1 -> y data
        super(Dataset, self).__init__()

    def __len__(self):
        return len(self.X_list[0])

    def __getitem__(self, index):
        ret_ds = []
        for i in range(self.num_views):
            x = self.X_list[i][index]
            y = self.y_list[i][index]
            ret_ds.append((torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)))
        return ret_ds


def batch_collator(batch, prot_desc_dict, spec, cuda_prot=True):
    batch = np.array(batch)  # batch.shape structure: (batch_size, x-0/y-1/w-2 data, view index)
    data = {}
    # num_active_views = reduce(lambda x1, x2: x1 + x2, flags.values())
    funcs = {
        "ecfp4": process_ecfp_view_data,
        "ecfp8": process_ecfp_view_data,
        "weave": process_weave_view_data,
        "gconv": process_gconv_view_data,
        "gnn": process_gnn_view_data
    }
    active_views = []
    if isinstance(spec, dict):
        for k in spec:
            if spec[k]:
                active_views.append(k)
    else:
        active_views.append(spec)
    for i, v_name in enumerate(active_views):
        func = funcs[v_name]
        data[v_name] = (func(batch, prot_desc_dict, i, cuda_prot), batch[:, 1, i], batch[:, 2, i])
    return len(batch), data


def process_ecfp_view_data(X, prot_desc_dict, idx, cuda_prot):
    """
    Converts ECFP-Protein pair dataset to a pytorch tensor.

    :param X:
    :param prot_desc_dict:
    :return:
    """
    mols_tensor = prots_tensor = None
    prot_names = None
    x_data = None
    if X is not None:
        x_data = X[:, 0, idx]
        mols = [pair[0] for pair in x_data]
        mols_tensor = torch.from_numpy(np.array([mol.get_array() for mol in mols]))
        prots = [pair[1] for pair in x_data]
        prot_names = [prot.get_name() for prot in prots]
        prot_desc = [prot_desc_dict[prot_name] for prot_name in prot_names]
        prot_desc = np.array(prot_desc)
        prot_desc = prot_desc.reshape(prot_desc.shape[0], prot_desc.shape[2])
        prots_tensor = torch.from_numpy(prot_desc)
    return cuda(mols_tensor.float()), cuda(
        prots_tensor.float()) if cuda_prot else prots_tensor.float(), prot_names, x_data


def process_weave_view_data(X, prot_desc_dict, idx, cuda_prot):
    """
    Converts Weave-Protein pair dataset to a pytorch tensor.

    :param X:
    :param prot_desc_dict:
    :return:
    """
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    prot_descriptor = []
    n_atoms_list = []
    start = 0
    x_data = X[:, 0, idx]
    prot_names = []
    for im, pair in enumerate(x_data):
        mol, prot = pair
        n_atoms = mol.get_num_atoms()
        n_atoms_list.append(n_atoms)
        prot_names.append(prot.get_name())

        # number of atoms in each molecule
        atom_split.extend([im] * n_atoms)

        # index of pair features
        C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
        atom_to_pair.append(
            np.transpose(
                np.array([C1.flatten() + start,
                          C0.flatten() + start])))
        # number of pairs for each atom
        pair_split.extend(C1.flatten() + start)
        start = start + n_atoms

        # atom features
        atom_feat.append(mol.get_atom_features())
        # pair features
        n_pair_feat = mol.pairs.shape[2]
        pair_feat.append(
            np.reshape(mol.get_pair_features(),
                       (n_atoms * n_atoms, n_pair_feat)))
        prot_descriptor.append(prot_desc_dict[prot.get_name()])
    prots_tensor = torch.from_numpy(np.concatenate(prot_descriptor, axis=0))
    mol_data = [
        cuda(torch.tensor(np.concatenate(atom_feat, axis=0), dtype=torch.float)),
        cuda(torch.tensor(np.concatenate(pair_feat, axis=0), dtype=torch.float)),
        cuda(torch.tensor(np.array(pair_split), dtype=torch.int)),
        cuda(torch.tensor(np.concatenate(atom_to_pair, axis=0), dtype=torch.long)),
        cuda(torch.tensor(np.array(atom_split), dtype=torch.int)),
        n_atoms_list
    ]
    return mol_data, cuda(prots_tensor.float()) if cuda_prot else prots_tensor.float(), prot_names, x_data


def process_gconv_view_data(X, prot_desc_dict, idx, cuda_prot):
    """
    Converts Graph convolution-Protein pair dataset to a pytorch tensor.

    :param X:
    :param prot_desc_dict:
    :return:
    """
    mol_data = []
    n_atoms_list = []
    x_data = X[:, 0, idx]
    mols = []
    for pair in x_data:
        mol, prot = pair
        n_atoms = mol.get_num_atoms()
        n_atoms_list.append(n_atoms)
        mols.append(mol)

    multiConvMol = ConvMol.agglomerate_mols(mols)
    mol_data.append(cuda(torch.from_numpy(multiConvMol.get_atom_features())))
    mol_data.append(cuda(torch.from_numpy(multiConvMol.deg_slice)))
    mol_data.append(cuda(torch.tensor(multiConvMol.membership)))
    mol_data.append(n_atoms_list)
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        mol_data.append(cuda(torch.from_numpy(multiConvMol.get_deg_adjacency_lists()[i])))

    # protein
    prots = [pair[1] for pair in x_data]
    prot_names = [prot.get_name() for prot in prots]
    prot_desc = [prot_desc_dict[prot_name] for prot_name in prot_names]
    prot_desc = np.array(prot_desc)
    prot_desc = prot_desc.reshape(prot_desc.shape[0], prot_desc.shape[2])
    prots_tensor = cuda(torch.from_numpy(prot_desc)) if cuda_prot else torch.from_numpy(prot_desc)

    return mol_data, prots_tensor.float(), prot_names, x_data


def process_gnn_view_data(X, prot_desc_dict, idx, cuda_prot):
    prot_names = []
    x_data = X[:, 0, idx]
    adjacency_matrices = []
    fp_profiles = []
    prot_desc = []

    for pair in x_data:
        mol, prot = pair
        adjacency_matrices.append(torch.from_numpy(mol.adjacency).float())
        fp_profiles.append(cuda(torch.tensor(mol.fingerprints, dtype=torch.long)))
        prot_names.append(prot.get_name())
        prot_desc.append(prot_desc_dict[prot.get_name()])

    # compound
    adjacency_matrices = block_diag_irregular(adjacency_matrices)
    axis = [len(f) for f in fp_profiles]
    M = np.concatenate([np.repeat(len(f), len(f)) for f in fp_profiles])
    M = torch.unsqueeze(torch.FloatTensor(M), 1)
    fingerprints = torch.cat(fp_profiles)
    mol_data = (fingerprints, cuda(adjacency_matrices), cuda(M), axis)

    # protein - PSC
    prot_desc = np.array(prot_desc)
    prot_desc = prot_desc.reshape(prot_desc.shape[0], prot_desc.shape[2])
    prots_tensor = cuda(torch.from_numpy(prot_desc)) if cuda_prot else torch.from_numpy(prot_desc)
    return mol_data, prots_tensor.float(), prot_names, x_data


# def pad(matrices, pad_value=0):
#     """Pad adjacency matrices for batch processing."""
#     sizes = [m.shape[0] for m in matrices]
#     M = sum(sizes)
#     pad_matrices = pad_value + np.zeros((M, M))
#     i = 0
#     for j, m in enumerate(matrices):
#         j = sizes[j]
#         pad_matrices[i:i + j, i:i + j] = m
#         i += j
#     return cuda(torch.FloatTensor(pad_matrices))


def cuda(tensor):
    if _cuda:
        return tensor.cuda()
    else:
        return tensor


def get_data(featurizer, flags, prot_sequences, seed, mf_simboost_data_dict=None):
    # logger = get_logger(name="Data loader")
    print("--------------About to load {}-{} data-------------".format(featurizer, flags['dataset_name']))
    data = load_dti_data(featurizer=featurizer,
                         dataset_name=flags['dataset_name'],
                         dataset_file=flags['dataset_file'],
                         prot_seq_dict=prot_sequences,
                         input_protein=True,
                         cross_validation=flags['cv'],
                         test=flags['test'],
                         fold_num=flags['fold_num'],
                         split=flags['splitting_alg'],
                         reload=flags['reload'],
                         predict_cold=flags['predict_cold'],
                         cold_drug=flags['cold_drug'],
                         cold_target=flags['cold_target'],
                         cold_drug_cluster=flags['cold_drug_cluster'],
                         split_warm=flags['split_warm'],
                         mode='regression' if not hasattr(flags, 'mode') else flags['mode'],
                         seed=seed,
                         filter_threshold=flags["filter_threshold"],
                         mf_simboost_data_dict=mf_simboost_data_dict)
    print("--------------{}-{} data loaded-------------".format(featurizer, flags['dataset_name']))
    return data


def compute_similarity_kernel_matrices(dataset):
    """
    Computes the drug-drug and protein-protein kernel matrices for kernel-based methods (e.g. Kron-RLS)

    :param dataset:
    :return: tuple
    """
    start = time.time()
    print("Computing kernel matrices (KD_dict, KT_dict)")
    all_comps = set()
    all_prots = set()
    for idx, pair in enumerate(dataset.X):
        mol, prot = pair
        all_comps.add(mol)
        all_prots.add(prot)

    # compounds / drugs
    comps_mat = {}
    for c1 in all_comps:
        fp1 = c1.fingerprint
        for c2 in all_comps:
            fp2 = c2.fingerprint
            # Tanimoto coefficient
            score = DataStructs.TanimotoSimilarity(fp1, fp2)
            comps_mat[Pair(c1, c2)] = score

    # proteins / targets
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'  # SW algorithm
    prots_mat = {}
    for p1 in all_prots:
        seq1 = p1.sequence[1]
        p1_score = aligner.score(seq1, seq1)
        for p2 in all_prots:
            seq2 = p2.sequence[1]
            p2_score = aligner.score(seq2, seq2)
            score = aligner.score(seq1, seq2)
            # Normalized SW score
            prots_mat[Pair(p1, p2)] = score / (sqrt(p1_score) * sqrt(p2_score))

    print("Kernel entities: Drugs={}, Prots={}".format(len(all_comps), len(all_prots)))
    duration = time.time() - start
    print("Kernel matrices computation finished in: {:.0f}m {:.0f}s".format(duration // 60, duration % 60))
    return comps_mat, prots_mat


def compute_simboost_drug_target_features(dataset, mf_simboost_data_dict, nbins=10, sim_threshold=0.5):
    """
    Constructs the type 1,2, and 3 features (with the matrix factorization part) of SimBoost as described in:
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z
    The Matrix Factorization part is deferred to the mf.py script.

    :param sim_threshold:
    :param nbins:
    :param dataset:
    :return:
    """
    assert isinstance(mf_simboost_data_dict, dict), "Drug-Target features dictionary must be provided."
    print('SimBoost Drug-Target feature vector computation started')

    print('Processing M matrix')
    all_comps = set()
    all_prots = set()
    pair_to_value_y = {}
    Mgraph = nx.Graph(name='drug_target_network')
    Mrows = defaultdict(list)
    Mcols = defaultdict(list)
    for x, y, w, id in tqdm(dataset.itersamples()):
        mol, prot = x
        all_comps.add(mol)
        all_prots.add(prot)
        pair_to_value_y[Pair(mol, prot)] = y
        Mrows[mol].append(y)
        Mcols[prot].append(y)
        Mgraph.add_edge(mol, prot, weight=y)
    print('Number of compounds = %d' % len(all_comps))
    print('Number of targets = %d' % len(all_prots))

    # compounds / drugs
    print('Processing drug similarity matrix')
    D = {}
    Dgraph = nx.Graph(name='drug_drug_network')
    for c1 in tqdm(all_comps):
        fp1 = c1.fingerprint
        for c2 in all_comps:
            fp2 = c2.fingerprint
            # Tanimoto coefficient
            score = DataStructs.TanimotoSimilarity(fp1, fp2)
            D[Pair(c1, c2)] = score
            Dgraph.add_nodes_from([c1, c2])
            if score >= sim_threshold and c1 != c2:
                Dgraph.add_edge(c1, c2)
    comp_feats = compute_type2_features(compute_type1_features(Mrows, all_comps, D, nbins), D, Dgraph)

    # proteins / targets
    print('Processing target similarity matrix')
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'  # SW algorithm
    T = {}
    Tgraph = nx.Graph(name='target_target_network')
    for p1 in tqdm(all_prots):
        seq1 = p1.sequence[1]
        p1_score = aligner.score(seq1, seq1)
        for p2 in all_prots:
            seq2 = p2.sequence[1]
            p2_score = aligner.score(seq2, seq2)
            score = aligner.score(seq1, seq2)
            # Normalized SW score
            normalized_score = score / (sqrt(p1_score) * sqrt(p2_score))
            T[Pair(p1, p2)] = normalized_score
            Tgraph.add_nodes_from([p1, p2])
            if normalized_score >= sim_threshold and p1 != p2:
                Tgraph.add_edge(p1, p2)
    prot_feats = compute_type2_features(compute_type1_features(Mcols, all_prots, T, nbins), T, Tgraph)

    pbar = UnboundedProgressbar()
    pbar.start()

    print('Processing type 3 features')
    # Type 3 features
    btw_cent = nx.betweenness_centrality(Mgraph)
    cls_cent = nx.closeness_centrality(Mgraph)
    # eig_cent = nx.eigenvector_centrality(Mgraph, tol=1e-3, max_iter=500)
    # pagerank = nx.pagerank(Mgraph, tol=1e-3, max_iter=1000)
    drug_target_feats_dict = defaultdict(list)
    vec_lengths = []

    # Retrieve data from the Matrix Factorization stage
    comp_mat = mf_simboost_data_dict['comp_mat']
    prot_mat = mf_simboost_data_dict['prot_mat']
    comp_index = mf_simboost_data_dict['comp_index']
    prot_index = mf_simboost_data_dict['prot_index']
    for pair in tqdm(pair_to_value_y):
        comp, prot = pair.p1, pair.p2
        feat = drug_target_feats_dict[Pair(comp, prot)]
        # mf
        cidx = comp_index[comp]
        pidx = prot_index[prot]
        c_vec = comp_mat[cidx].tolist()
        p_vec = prot_mat[pidx].tolist()
        mf = c_vec + p_vec
        feat.extend(mf)

        # d.t.ave
        d_av_lst = []
        for n in Mgraph.neighbors(prot):
            if Pair(comp, n) in pair_to_value_y:
                d_av_lst.append(pair_to_value_y[Pair(comp, n)])
        if len(d_av_lst) > 0:
            feat.append(np.mean(d_av_lst))

        # t.d.ave
        t_av_lst = []
        for n in Mgraph.neighbors(comp):
            if Pair(n, prot) in pair_to_value_y:
                t_av_lst.append(pair_to_value_y[Pair(n, prot)])
        if len(t_av_lst) > 0:
            feat.append(np.mean(t_av_lst))

        # d.t.bt, d.t.cl, d.t.ev
        feat.append(btw_cent[comp])
        feat.append(btw_cent[prot])
        feat.append(cls_cent[comp])
        feat.append(cls_cent[prot])
        # feat.append(eig_cent[comp])
        # feat.append(eig_cent[prot])

        # d.t.pr
        # feat.append(pagerank[comp])
        # feat.append(pagerank[prot])

        # add type 1 features
        feat.extend(comp_feats[comp])
        feat.extend(prot_feats[prot])

        vec_lengths.append(len(feat))

    # zero-pad all vectors to be of the same dimension
    dim = max(vec_lengths)
    for k in drug_target_feats_dict:
        feat = drug_target_feats_dict[k]
        pvec = [0] * (dim - len(feat))
        feat.extend(pvec)

    pbar.stop()
    pbar.join()
    print('SimBoost Drug-Target feature vector computation finished. Vector dimension={}'.format(dim))
    return drug_target_feats_dict


def compute_type1_features(M, all_E, Edict, nbins):
    """
    Computes type 1 features of a set of entities (E)
    :param M:
    :param Edict:
    :param nbins:
    :return:
        A dict of entity-feature elements
    """
    feats_dict = defaultdict(list)
    for entity in all_E:
        feat = feats_dict[entity]
        # n.obs
        feat.append(len(M[entity]))

        # ave.sim
        sim_scores = [Edict[Pair(entity, entity2)] for entity2 in all_E]
        feat.append(np.mean(sim_scores))

        # hist.sim
        hist = np.histogram(sim_scores, bins=nbins)[0]
        feat.extend(hist.tolist())

        # ave.val in M
        feat.append(np.mean(M[entity]))
    return feats_dict


def compute_type2_features(type1_feats_dict, Edict, Egraph):
    """
    Computes type 2 features of a set of entities whose type 1 features have already been computed.
    :param type1_feats_dict:
    :param Edict:
    :param Egraph:
    :return:
        A dict of entity-feature elements
    """
    feats_dict = defaultdict(list)
    btw_cent = nx.betweenness_centrality(Egraph)
    cls_cent = nx.closeness_centrality(Egraph)
    eig_cent = nx.eigenvector_centrality(Egraph, tol=1e-5, max_iter=200)
    pagerank = nx.pagerank(Egraph)
    for node in Egraph.nodes():
        feat = feats_dict[node]

        # num.nb
        neighbors = list(Egraph.neighbors(node))
        feat.append(len(neighbors))

        # k.sim
        neighbors_sim_score = [Edict[Pair(node, neighbor)] for neighbor in neighbors]
        feat.extend(neighbors_sim_score)

        if len(neighbors) > 0:
            # k.ave.feat
            neighs_t1_feat = np.array([type1_feats_dict[neighbor] for neighbor in neighbors])
            avg_neighs_t1_feat = np.mean(neighs_t1_feat, axis=1)
            feat.extend(avg_neighs_t1_feat.tolist())

            # k.w.ave.feat
            w_ave_feat = np.array(neighbors_sim_score) * avg_neighs_t1_feat
            feat.extend(w_ave_feat.tolist())

    # bt, cl, ev
    feat.append(btw_cent[node])
    feat.append(cls_cent[node])
    feat.append(eig_cent[node])

    # pr
    feat.append(pagerank[node])
    return feats_dict


class Pair(object):
    """
    Order-invariant pair.
    """

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, other):
        assert isinstance(other, Pair)
        return (self.p1 == other.p1 and self.p2 == other.p2) or (self.p1 == other.p2 and self.p2 == other.p1)

    def __hash__(self):
        h1, h2 = hash(self.p1), hash(self.p2)
        return hash('{}-{}'.format(min(h1, h2), max(h1, h2)))


def featurize_datasets(jova_args, feat_dict, flags, prot_seq_dict, seeds):
    """
    Ensures all possible compound views that would be used are featurized for every seed.

    :param feat_dict:
    :param flags:
    :param seeds:
    :param prot_seq_dict:
    :param jova_args:
    :return:
    """
    for seed in seeds:
        comp_views = set()
        for arg in jova_args:
            dummy = ViewsReg()
            dummy.parse_views(arg)
            for cv in dummy.c_views:
                comp_views.add(cv)
        for v in comp_views:
            d = get_data(feat_dict[v], flags, prot_sequences=prot_seq_dict, seed=seed)
            del d


def compute_train_val_test_kronrls_mats(train, val, test, Kd_dict, Kt_dict):
    """

    :param train:
    :param val:
    :param test:
    :param Kd_dict:
    :param Kt_dict:
    :return:
    """
    # Construct KD and KT
    train_mol = set()
    train_prot = set()
    labels = defaultdict(lambda: np.nan)
    weights = defaultdict(float)
    for x, y, w, _ in train.itersamples():
        mol, prot = x
        train_mol.add(mol)
        train_prot.add(prot)
        labels[Pair(mol, prot)] = float(y)
        weights[Pair(mol, prot)] = float(w)
    KD = np.array([[Kd_dict[Pair(c1, c2)] for c2 in train_mol] for c1 in train_mol], dtype=np.float)
    KT = np.array([[Kt_dict[Pair(p1, p2)] for p2 in train_prot] for p1 in train_prot], dtype=np.float)
    Y = np.array([[labels[Pair(c, p)] for p in train_prot] for c in train_mol], dtype=np.float)
    W = np.array([[weights[Pair(c, p)] for p in train_prot] for c in train_mol], dtype=np.float)
    # print('KD.shape={}, KT.shape={}, Y.shape={}, W.shape={}'.format(KD.shape, KT.shape, Y.shape, W.shape))

    # imputation
    rows, cols = np.where(np.isnan(Y) == False)
    Y_knowns = Y[rows, cols]
    impvalue = np.mean(Y_knowns)
    rows, cols = np.where(np.isnan(Y) == True)
    Y[rows, cols] = impvalue

    # Construct validation set matrix
    val_mol = set()
    val_prot = set()
    labels_val = defaultdict(float)
    weights_val = defaultdict(float)
    for x, y, w, _ in val.itersamples():
        dx, tx = x
        val_mol.add(dx)
        val_prot.add(tx)
        labels_val[Pair(dx, tx)] = float(y)
        weights_val[Pair(dx, tx)] = float(w)
    KD_val = np.array([[Kd_dict[Pair(c1, c2)] for c2 in train_mol] for c1 in val_mol], dtype=np.float)
    KT_val = np.array([[Kt_dict[Pair(p1, p2)] for p2 in train_prot] for p1 in val_prot], dtype=np.float)
    Y_val = np.array([[labels_val[Pair(c, p)] for p in val_prot] for c in val_mol], dtype=np.float)
    W_val = np.array([[weights_val[Pair(c, p)] for p in val_prot] for c in val_mol], dtype=np.float)

    # Construct test set matrix
    test_mol = set()
    test_prot = set()
    labels_test = defaultdict(float)
    weights_test = defaultdict(float)
    for x, y, w, _ in test.itersamples():
        dx, tx = x
        test_mol.add(dx)
        test_prot.add(tx)
        labels_test[Pair(dx, tx)] = float(y)
        weights_test[Pair(dx, tx)] = float(w)
    KD_test = np.array([[Kd_dict[Pair(c1, c2)] for c2 in train_mol] for c1 in test_mol], dtype=np.float)
    KT_test = np.array([[Kt_dict[Pair(p1, p2)] for p2 in train_prot] for p1 in test_prot], dtype=np.float)
    Y_test = np.array([[labels_test[Pair(c, p)] for p in test_prot] for c in test_mol], dtype=np.float)
    W_test = np.array([[weights_test[Pair(c, p)] for p in test_prot] for c in test_mol], dtype=np.float)

    return {'train': (KD, KT, Y, W), 'val': (KD_val, KT_val, Y_val, W_val), 'test': (KD_test, KT_test, Y_test, W_test)}


def compute_MF_entities_matrix(dataset):
    """
    Computes the label matrix (M) for the Matrix Factorization stage of SimBoost.

    :param dataset:
    :return:
    """
    all_comps = set()
    all_prots = set()
    pair_to_value_y = defaultdict(lambda: float())
    for x, y, w, id in dataset.itersamples():
        mol, prot = x
        all_comps.add(mol)
        all_prots.add(prot)
        pair_to_value_y[Pair(mol, prot)] = y

    M = torch.zeros((len(all_comps), len(all_prots)))
    # Construct matrix
    for i, c in enumerate(all_comps):
        for j, p in enumerate(all_prots):
            M[i, j] = float(pair_to_value_y[Pair(c, p)])
    return {'labels': M, 'compounds': all_comps, 'proteins': all_prots}
