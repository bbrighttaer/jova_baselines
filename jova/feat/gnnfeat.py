# Re-organization of the original GNN code of in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
# Author: bbrighttaer
# Project: jova
# Date: 10/29/19
# Time: 2:12 AM
# File: gnnfeat.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
from collections import defaultdict

import numpy as np
from rdkit import Chem

from jova.feat import Featurizer

__all__ = ['GNNFeaturizer', 'GnnMol']


def create_atoms(feat, mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [feat.atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(feat, mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = feat.bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(feat, atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [feat.fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(feat.fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = feat.edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


class GnnMol(object):

    def __init__(self, mol, fingerprints, adjacency, smiles):
        self.mol = mol
        self.fingerprints = fingerprints
        self.adjacency = adjacency
        self.smiles = smiles

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            assert self.smiles is not None
            return self.smiles == other.smiles
        return False

    def __hash__(self):
        assert self.smiles is not None
        return hash(self.smiles)


class GNNFeaturizer(Featurizer):
    """
    Graph Neural Net.

    Compound featurizer described in https://academic.oup.com/bioinformatics/article/35/2/309/5050020
    This is basically a featurization wrapper of the initial GNN code accompanying the work cited above.
    """

    name = ["gnn_mol"]

    def __init__(self, radius=2):
        super(GNNFeaturizer, self).__init__()
        self.radius = radius
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))

    def _featurize(self, mol, smiles):
        """
        Featurizes a compound as described in the paper cited above.
        :param mol:
        :param smiles:
        :return:
        """
        mol = Chem.AddHs(mol)  # Consider hydrogens.
        # Process each fragment in the compound separately and join the fingerprints of all fragments to form the
        # fingerprint of the compound/molecule.
        # We think this provides a better handling of SMILES with '.' in them (Disconnected structures)
        # The original codes of the aforecited paper removes all such samples.
        fragments = Chem.GetMolFrags(mol, asMols=True)
        frag_fingerprints = []
        for frag_mol in fragments:
            atoms = create_atoms(self, frag_mol)
            i_jbond_dict = create_ijbonddict(self, frag_mol)
            fingerprints = extract_fingerprints(self, atoms, i_jbond_dict, self.radius)
            frag_fingerprints.append(fingerprints)
        fingerprints = np.concatenate(frag_fingerprints)
        adjacency = create_adjacency(mol)
        return GnnMol(mol, fingerprints, adjacency, smiles)

    def save_featurization_info(self, save_dir):
        """
        Persists GNN featurization data needed at runtime.

        :param save_dir: folder to save objects.
        :return:
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'fingerprint_dict.pickle'), 'wb') as f:
            pickle.dump(dict(self.fingerprint_dict), f)

# if __name__ == '__main__':
#     #smiles_str = 'CC(C)(C)OC(=O)N1CCC(CC1)n2ncc3c(nc(nc23)c4ccc(N)cc4)N5CCOCC5'
#     smiles_str = '[Na+].O\\N=C/1\\C(=C/2\\C(=O)Nc3ccc(cc23)S(=O)(=O)[O-])\\Nc4ccccc14'
#     mol = Chem.MolFromSmiles(smiles_str)
#     print(f'Number of atoms={mol.GetNumAtoms()}', f'Number of bonds={mol.GetNumBonds()}')
#     feat = GNNFeaturizer()
#     feat.featurize([mol], smiles_str)
