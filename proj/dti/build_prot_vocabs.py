# Author: bbrighttaer
# Project: jova
# Date: 10/17/19
# Time: 11:24 PM
# File: build_prot_vocabs.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from tqdm import tqdm


def split_sequence(sequence, ngram, vocab_dict):
    sequence = '-' + sequence + '='
    words = [vocab_dict[sequence[i:i + ngram]]
             for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def create_words(sequence, ngram=3, offsets=(0,)):
    all_words = []
    for offset in offsets:
        words = [sequence[i:i + ngram]
                 for i in range(offset, len(sequence) - offset - ngram + 1)]
        all_words += words
    return all_words


def split_sequence_overlapping(sequence, ngram=3):
    words = [sequence[i:i + ngram]
             for i in range(len(sequence) - ngram + 1)]
    return words


def create_protein_profile(vocab, words, window):
    profile = group_ngrams([vocab[w] for w in words], window, len(vocab))
    return np.array(profile)


def dump_binary(obj, filename, clazz):
    with open(filename, 'wb') as f:
        pickle.dump(clazz(obj), f)


def load_binary(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def group_ngrams(p_words, window, fill_value):
    """

    :param p_words:
    :param window:
    :param fill_value: Index for retrieving all zeros for padded regions. When loading the pre-trained
    embedding matrix, this row is filled with zeros. This is set as an additional row (last row) of the matrix.
    :return:
    """
    w = []
    start = 0
    for i in range(len(p_words) // window + 1):
        grouped_words = p_words[start:start + window]
        grouped_words = grouped_words + [fill_value] * (window - len(grouped_words))
        w.append(grouped_words)
        start += window
    return w


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a protein dictionary for training embeddings.")

    parser.add_argument('--prot_desc_path',
                        dest='prot_files',
                        action='append',
                        required=True,
                        help='A list containing paths to protein descriptors.')
    parser.add_argument('--ngram',
                        type=int,
                        default=3,
                        help='Length of each segment')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Prints every entry being processed')
    parser.add_argument('--vocab',
                        type=str,
                        required=True,
                        help='The file containing protein words (keys) and their index (values) '
                             'in the ProtVec embeddings')
    parser.add_argument('--window',
                        type=int,
                        default=11,
                        help='The window for forming protein sub-sequences from the n-grams')
    # parser.add_argument('--dim',
    #                     type=int,
    #                     required=True,
    #                     help='The dimension of the embeddings for each n-gram/word in the protein')
    args = parser.parse_args()

    word_dict = load_binary(args.vocab)
    proteins = {}
    print("Window: {}".format(args.window))

    for file in args.prot_files:
        print("Loading %s" % file)
        df = pd.read_csv(file)
        for row in tqdm(df.itertuples()):
            label = row[1]
            sequence = row[2]
            if args.verbose:
                print("Label={}, Sequence={}".format(label, sequence))
            words = split_sequence_overlapping(sequence, args.ngram)
            protein_profile = create_protein_profile(word_dict, words, args.window)
            proteins[label] = protein_profile
    print("Saving files...")
    dump_binary(proteins, '../../data/protein/proteins.profile', dict)
    # dump_binary(word_dict, '../../data/protein/proteins.vocab', dict)
    print("Info: vocab size={}, protein profiles saved={}".format(len(word_dict), len(proteins)))
