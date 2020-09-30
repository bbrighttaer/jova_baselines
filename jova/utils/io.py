"""
Modified by bbrighttaer
Original source taken from: https://github.com/simonfqy/PADME/blob/master/dcCustom/utils/save.py and I guess DeepChem.
Simple utils to save and load from disk.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import json
import logging
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
# from sklearn.externals import joblib as old_joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import jova


def log(string, verbose=True):
    """Print string if verbose."""
    if verbose:
        print(string)


def get_logger(name=None, level='INFO', stream='stderr', filename=None, log_dir='./logs/'):
    """
    Creates and return a logger to both console and a specified file.

    :param log_dir: The directory of the log file
    :param filename: The file to be logged into. It shall be in ./logs/
    :param name: The name of the logger
    :param level: The logging level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL
    :return: The created logger
    :param stream: Either 'stderr' or 'stdout'
    """
    os.makedirs(log_dir, exist_ok=True)
    stream = sys.stderr if stream == 'stderr' else sys.stdout
    log_level = {'DEBUG': logging.DEBUG,
                 'INFO': logging.INFO,
                 'WARNING': logging.WARNING,
                 'ERROR': logging.ERROR,
                 'CRITICAL': logging.CRITICAL}.get(level.upper(), 'INFO')
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(os.path.join(log_dir, filename + '.log')))
    if stream:
        handlers.append(logging.StreamHandler(stream))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=handlers)
    return logging.getLogger(name)


def save_to_disk(dataset, filename, compress=3):
    """Save a dataset to file."""
    joblib.dump(dataset, filename, compress=compress)


def get_input_type(input_file):
    """Get type of input file. Must be csv/pkl.gz/sdf file."""
    filename, file_extension = os.path.splitext(input_file)
    # If gzipped, need to compute extension again
    if file_extension == ".gz":
        filename, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
        return "csv"
    elif file_extension == ".pkl":
        return "pandas-pickle"
    elif file_extension == ".joblib":
        return "pandas-joblib"
    elif file_extension == ".sdf":
        return "sdf"
    else:
        raise ValueError("Unrecognized extension %s" % file_extension)


def load_data(input_files, shard_size=None, verbose=True):
    """Loads data from disk.

    For CSV files, supports sharded loading for large files.
    """
    if not len(input_files):
        return
    input_type = get_input_type(input_files[0])
    if input_type == "sdf":
        if shard_size is not None:
            log("Ignoring shard_size for sdf input.", verbose)
        for value in load_sdf_files(input_files):
            yield value
    elif input_type == "csv":
        for value in load_csv_files(input_files, shard_size, verbose=verbose):
            yield value
    elif input_type == "pandas-pickle":
        for input_file in input_files:
            yield load_pickle_from_disk(input_file)


def load_sdf_files(input_files, clean_mols):
    """Load SDF file into dataframe."""
    dataframes = []
    for input_file in input_files:
        # Tasks are stored in .sdf.csv file
        raw_df = next(load_csv_files([input_file + ".csv"], shard_size=None))
        # Structures are stored in .sdf file
        print("Reading structures from %s." % input_file)
        suppl = Chem.SDMolSupplier(str(input_file), clean_mols, False, False)
        df_rows = []
        for ind, mol in enumerate(suppl):
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                df_rows.append([ind, smiles, mol])
        mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
        dataframes.append(pd.concat([mol_df, raw_df], axis=1, join='inner'))
    return dataframes


def load_csv_files(filenames, shard_size=None, verbose=True):
    """Load data as pandas dataframe."""
    # First line of user-specified CSV *must* be header.
    shard_num = 1
    for filename in filenames:
        if shard_size is None:
            yield pd.read_csv(filename)
        else:
            log("About to start loading CSV from %s" % filename, verbose)
            for df in pd.read_csv(filename, chunksize=shard_size):
                log("Loading shard %d of size %s." % (shard_num, str(shard_size)),
                    verbose)
                df = df.replace(np.nan, str(""), regex=True)
                shard_num += 1
                yield df


def seq_one_hot_encode(sequences):
    """One hot encodes list of genomic sequences.

    Sequences encoded have shape (N_sequences, 4, sequence_length, 1).
    Here 4 is for the 4 basepairs (ACGT) present in genomic sequences.
    These sequences will be processed as images with one color channel.

    Parameters
    ----------
    sequences: np.ndarray
      Array of genetic sequences

    Raises
    ------
    ValueError:
      If sequences are of different lengths.

    Returns
    -------
    np.ndarray: Shape (N_sequences, 4, sequence_length, 1).
    """
    sequence_length = len(sequences[0])
    # depends on Python version
    integer_type = np.int32
    # The label encoder is given characters for ACGTN
    label_encoder = LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type))
    # These are transformed in 0, 1, 2, 3, 4 in input sequence
    integer_array = []
    # TODO(rbharath): Unlike the DRAGONN implementation from which this
    # was ported, I couldn't transform the "ACGT..." strings into
    # integers all at once. Had to do one at a time. Might be worth
    # figuring out what's going on under the hood.
    for sequence in sequences:
        if len(sequence) != sequence_length:
            raise ValueError("All sequences must be of same length")
        integer_seq = label_encoder.transform(
            np.array((sequence,)).view(integer_type))
        integer_array.append(integer_seq)
    integer_array = np.concatenate(integer_array)
    integer_array = integer_array.reshape(len(sequences), sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse=False, n_values=5, dtype=integer_type).fit_transform(integer_array)

    return one_hot_encoding.reshape(len(sequences), sequence_length, 5,
                                    1).swapaxes(1, 2)[:, [0, 1, 2, 4], :, :]


def encode_fasta_sequence(fname):
    """
    Loads fasta file and returns an array of one-hot sequences.

    Parameters
    ----------
    fname: str
      Filename of fasta file.
    """
    name, seq_chars = None, []
    sequences = []
    with open(fname) as fp:
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    sequences.append(''.join(seq_chars).upper())
                name, seq_chars = line, []
            else:
                seq_chars.append(line)
    if name is not None:
        sequences.append(''.join(seq_chars).upper())

    return seq_one_hot_encode(np.array(sequences))


def save_metadata(tasks, metadata_df, data_dir):
    """
    Saves the metadata for a DiskDataset
    Parameters
    ----------
    tasks: list of str
      Tasks of DiskDataset
    metadata_df: pd.DataFrame
    data_dir: str
      Directory to store metadata
    Returns
    -------
    """
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
    tasks_filename = os.path.join(data_dir, "tasks.json")
    with open(tasks_filename, 'w') as fout:
        json.dump(tasks, fout)
    metadata_df.to_csv(metadata_filename, index=False, compression='gzip')


def load_from_disk(filename):
    """Load a dataset from file."""
    name = filename
    if os.path.splitext(name)[1] == ".gz":
        name = os.path.splitext(name)[0]
    if os.path.splitext(name)[1] == ".pkl":
        return load_pickle_from_disk(filename)
    elif os.path.splitext(name)[1] == ".joblib":
        return joblib.load(filename)
        # try:
        #     return joblib.load(filename)
        # except KeyError:
        #     # Try older joblib version for legacy files.
        #     return old_joblib.load(filename)
        # except ValueError:
        #     return old_joblib.load(filename)
    elif os.path.splitext(name)[1] == ".csv":
        # First line of user-specified CSV *must* be header.
        df = pd.read_csv(filename, header=0)
        df = df.replace(np.nan, str(""), regex=True)
        return df
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def load_sharded_csv(filenames):
    """Load a dataset from multiple files. Each file MUST have same column headers"""
    dataframes = []
    for name in filenames:
        placeholder_name = name
        if os.path.splitext(name)[1] == ".gz":
            name = os.path.splitext(name)[0]
        if os.path.splitext(name)[1] == ".csv":
            # First line of user-specified CSV *must* be header.
            df = pd.read_csv(placeholder_name, header=0)
            df = df.replace(np.nan, str(""), regex=True)
            dataframes.append(df)
        else:
            raise ValueError("Unrecognized filetype for %s" % name)

    # combine dataframes
    combined_df = dataframes[0]
    for i in range(0, len(dataframes) - 1):
        combined_df = combined_df.append(dataframes[i + 1])
    combined_df = combined_df.reset_index(drop=True)
    return combined_df


def load_pickle_from_disk(filename):
    """Load dataset from pickle file."""
    if ".gz" in filename:
        with gzip.open(filename, "rb") as f:
            df = pickle.load(f)
    else:
        with open(filename, "rb") as f:
            df = pickle.load(f)
    return df


def save_cv_dataset_to_disk(save_dir, fold_dataset, fold_num, transformers):
    assert fold_num > 1
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        train_data = fold_dataset[i][0]
        valid_data = fold_dataset[i][1]
        train_data.move(train_dir)
        valid_data.move(valid_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), "wb") as f:
        pickle.dump(transformers, f)
    return None


def save_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name + ".mod")
    torch.save(model.state_dict(), file)
    # with open(os.path.join(path, "dummy_save.txt"), 'a') as f:
    #     f.write(name + '\n')


def save_dict_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name + ".pkl")
    with open(file, 'wb') as f:
        pickle.dump(dict(model), f)
    # with open(os.path.join(path, "dummy_save_dict.txt"), 'a') as f:
    #     f.write(name + '\n')


def load_dict_model(path, name):
    return load_pickle(os.path.join(path, name))


def load_model(path, name, dvc=None):
    """
    Loads the parameters of a model.

    :param path:
    :param name:
    :return: The saved state_dict.
    """
    if dvc is None:
        dvc = torch.device("cuda:0")
    return torch.load(os.path.join(path, name), map_location=dvc)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_numpy_array(file_name):
    return np.load(file_name, allow_pickle=True)


def save_numpy_array(array, path, name):
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name)
    np.save(file, array, allow_pickle=True)


def save_nested_cv_dataset_to_disk(save_dir, fold_dataset, fold_num, transformers, gnn_fingerprint, all_drug_sim_dict,
                                   all_prots_sim_dict, simboost_pairwise_feats_dict, mf_entities_dict):
    assert fold_num > 1
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        train_data = fold_dataset[i][0]
        valid_data = fold_dataset[i][1]
        test_data = fold_dataset[i][2]
        if train_data:
            train_data.move(train_dir)
        if valid_data:
            valid_data.move(valid_dir)
        if test_data:
            test_data.move(test_dir)

        # process kernel / kronrls data
        if all_drug_sim_dict and all_prots_sim_dict:
            save_kernel_data(fold_dataset[i][3], test_data, test_dir, train_data, train_dir, valid_data, valid_dir)

    with open(os.path.join(save_dir, "transformers.pkl"), "wb") as f:
        pickle.dump(transformers, f)
    if gnn_fingerprint is not None:
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "wb") as f:
            pickle.dump(dict(gnn_fingerprint), f)
    if all_drug_sim_dict is not None:
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(all_drug_sim_dict), f)
    if all_prots_sim_dict is not None:
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(all_prots_sim_dict), f)
    if simboost_pairwise_feats_dict is not None:
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "wb") as f:
            pickle.dump(dict(simboost_pairwise_feats_dict), f)
    if mf_entities_dict is not None:
        with open(os.path.join(save_dir, 'mf_entities_dict.pkl'), 'wb') as f:
            pickle.dump(dict(mf_entities_dict), f)
    return None


def save_kernel_data(kernel_data, test_data, test_dir, train_data, train_dir, valid_data, valid_dir):
    if train_data:
        kd, kt, Y, W = kernel_data['KD'], kernel_data['KT'], kernel_data['Y'], kernel_data['W']
        with open(os.path.join(train_dir, 'kd_train.pkl'), 'wb') as f:
            pickle.dump(kd, f)
        with open(os.path.join(train_dir, 'kt_train.pkl'), 'wb') as f:
            pickle.dump(kt, f)
        with open(os.path.join(train_dir, 'labels_mat_train.pkl'), 'wb') as f:
            pickle.dump(Y, f)
        with open(os.path.join(train_dir, 'weights_mat_train.pkl'), 'wb') as f:
            pickle.dump(W, f)
    if valid_data:
        kd_val, kt_val, Y_val, W_val = kernel_data['KD_val'], kernel_data['KT_val'], \
                                       kernel_data['Y_val'], kernel_data['W_val']
        with open(os.path.join(valid_dir, 'kd_val.pkl'), 'wb') as f:
            pickle.dump(kd_val, f)
        with open(os.path.join(valid_dir, 'kt_val.pkl'), 'wb') as f:
            pickle.dump(kt_val, f)
        with open(os.path.join(valid_dir, 'labels_mat_val.pkl'), 'wb') as f:
            pickle.dump(Y_val, f)
        with open(os.path.join(valid_dir, 'weights_mat_val.pkl'), 'wb') as f:
            pickle.dump(W_val, f)
    if test_data:
        kd_test, kt_test, Y_test, W_test = kernel_data['KD_test'], kernel_data['KT_test'], \
                                           kernel_data['Y_test'], kernel_data['W_test']
        with open(os.path.join(test_dir, 'kd_test.pkl'), 'wb') as f:
            pickle.dump(kd_test, f)
        with open(os.path.join(test_dir, 'kt_test.pkl'), 'wb') as f:
            pickle.dump(kt_test, f)
        with open(os.path.join(test_dir, 'labels_mat_test.pkl'), 'wb') as f:
            pickle.dump(Y_test, f)
        with open(os.path.join(test_dir, 'weights_mat_test.pkl'), 'wb') as f:
            pickle.dump(W_test, f)


def save_dataset_to_disk(save_dir, train, valid, test, transformers, gnn_fingerprint, drug_kernel_dict,
                         prot_kernel_dict, simboost_pairwise_feats_dict, kernel_data, mf_entities_dict):
    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if train:
        train.move(train_dir)
    if valid:
        valid.move(valid_dir)
    if test:
        test.move(test_dir)

    if kernel_data:
        save_kernel_data(kernel_data, test, test_dir, train, train_dir, valid, valid_dir)

    with open(os.path.join(save_dir, "transformers.pkl"), 'wb') as f:
        pickle.dump(transformers, f)
    if gnn_fingerprint is not None:
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "wb") as f:
            pickle.dump(dict(gnn_fingerprint), f)
    if drug_kernel_dict is not None:
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(drug_kernel_dict), f)
    if prot_kernel_dict is not None:
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "wb") as f:
            pickle.dump(dict(prot_kernel_dict), f)
    if simboost_pairwise_feats_dict is not None:
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "wb") as f:
            pickle.dump(dict(simboost_pairwise_feats_dict), f)
    if mf_entities_dict is not None:
        with open(os.path.join(save_dir, 'mf_entities_dict.pkl'), 'wb') as f:
            pickle.dump(dict(mf_entities_dict), f)
    return None


def load_nested_cv_dataset_from_disk(save_dir, fold_num):
    assert fold_num > 1
    train_data = []
    valid_data = []
    test_data = []
    kernel_data = []
    for i in range(fold_num):
        fold_dir = os.path.join(save_dir, "fold" + str(i + 1))
        train_dir = os.path.join(fold_dir, "train_dir")
        valid_dir = os.path.join(fold_dir, "valid_dir")
        test_dir = os.path.join(fold_dir, "test_dir")
        if not os.path.exists(train_dir):
            return False, None, list(), None, None, None, None
        train = jova.data.DiskDataset(train_dir)
        valid = jova.data.DiskDataset(valid_dir) if os.path.exists(valid_dir) else None
        test = jova.data.DiskDataset(test_dir) if os.path.exists(test_dir) else None
        train_data.append(train)
        valid_data.append(valid)
        test_data.append(test)

        # check for and load kernel data if present
        kernel_files = get_kernel_filepaths(test_dir, train_dir, valid_dir)
        for kfile in kernel_files:
            kfile_path = kernel_files[kfile]
            if not os.path.exists(kfile_path):
                continue
            with open(kfile_path, 'rb') as f:
                kernel_files[kfile] = pickle.load(f)
        kernel_data.append(kernel_files)

    gnn_fingerprint = None
    mf_entities_dict = None
    simboost_pairwise_feats_dict = all_drugs_sim_dict = all_prots_sim_dict = None

    if os.path.exists(os.path.join(save_dir, "gnn_fingerprint_dict.pkl")):
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "rb") as f:
            gnn_fingerprint = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "drug_drug_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "rb") as f:
            all_drugs_sim_dict = pickle.load(f)
    if os.path.exists(os.path.join(save_dir, "prot_prot_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "rb") as f:
            all_prots_sim_dict = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl")):
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "rb") as f:
            simboost_pairwise_feats_dict = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, 'mf_entities_dict.pkl')):
        mf_entities_dict = load_dict_model(save_dir, 'mf_entities_dict.pkl')

    loaded = True
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, list(zip(train_data, valid_data, test_data, kernel_data)), transformers, gnn_fingerprint, \
               (all_drugs_sim_dict, all_prots_sim_dict), simboost_pairwise_feats_dict, mf_entities_dict


def load_dataset_from_disk(save_dir):
    """
    Parameters
    ----------
    save_dir: str

    Returns
    -------
    loaded: bool
      Whether the load succeeded
    all_dataset: (dc.data.Dataset, dc.data.Dataset, dc.data.Dataset)
      The train, valid, test datasets
    transformers: list of dc.trans.Transformer
      The transformers used for this dataset

    """

    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if not os.path.exists(train_dir):
        return False, None, list(), None, None, None, None

    # check for and load kernel data if present
    kernel_files = get_kernel_filepaths(test_dir, train_dir, valid_dir)
    for kfile in kernel_files:
        kfile_path = kernel_files[kfile]
        if not os.path.exists(kfile_path):
            continue
        with open(kfile_path, 'rb') as f:
            kernel_files[kfile] = pickle.load(f)

    gnn_fingerprint = None
    mf_entities_dict = None
    simboost_pairwise_feats_dict = drug_sim_kernel_dict = prot_sim_kernel_dict = None

    if os.path.exists(os.path.join(save_dir, "gnn_fingerprint_dict.pkl")):
        with open(os.path.join(save_dir, "gnn_fingerprint_dict.pkl"), "rb") as f:
            gnn_fingerprint = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "drug_drug_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "drug_drug_kernel_dict.pkl"), "rb") as f:
            drug_sim_kernel_dict = pickle.load(f)
    if os.path.exists(os.path.join(save_dir, "prot_prot_kernel_dict.pkl")):
        with open(os.path.join(save_dir, "prot_prot_kernel_dict.pkl"), "rb") as f:
            prot_sim_kernel_dict = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl")):
        with open(os.path.join(save_dir, "simboost_pairwise_feats_dict.pkl"), "rb") as f:
            simboost_pairwise_feats_dict = pickle.load(f)

    if os.path.exists(os.path.join(save_dir, 'mf_entities_dict.pkl')):
        mf_entities_dict = load_dict_model(save_dir, 'mf_entities_dict.pkl')

    loaded = True
    train = jova.data.DiskDataset(train_dir)
    valid = jova.data.DiskDataset(valid_dir) if os.path.exists(valid_dir) else None
    test = jova.data.DiskDataset(test_dir) if os.path.exists(test_dir) else None
    all_dataset = (train, valid, test, kernel_files)
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        transformers = pickle.load(f)
        return loaded, all_dataset, transformers, gnn_fingerprint, \
               (drug_sim_kernel_dict, prot_sim_kernel_dict), simboost_pairwise_feats_dict, mf_entities_dict


def get_kernel_filepaths(test_dir, train_dir, valid_dir):
    return {'KD': os.path.join(train_dir, 'kd_train.pkl'),
            'KT': os.path.join(train_dir, 'kt_train.pkl'),
            'Y': os.path.join(train_dir, 'labels_mat_train.pkl'),
            'W': os.path.join(train_dir, 'weights_mat_train.pkl'),

            'KD_val': os.path.join(valid_dir, 'kd_val.pkl'),
            'KT_val': os.path.join(valid_dir, 'kt_val.pkl'),
            'Y_val': os.path.join(valid_dir, 'labels_mat_val.pkl'),
            'W_val': os.path.join(valid_dir, 'weights_mat_val.pkl'),

            'KD_test': os.path.join(test_dir, 'kd_test.pkl'),
            'KT_test': os.path.join(test_dir, 'kt_test.pkl'),
            'Y_test': os.path.join(test_dir, 'labels_mat_test.pkl'),
            'W_test': os.path.join(test_dir, 'weights_mat_test.pkl')
            }
