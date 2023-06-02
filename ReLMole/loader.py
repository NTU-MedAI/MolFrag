import os
import os.path as osp
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from itertools import repeat
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from chem import *


class MultiGraphData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 fg_x=None, fg_edge_index=None, atom2fg_index=None,
                 **kwargs):
        super(MultiGraphData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.fg_x = fg_x
        self.fg_edge_index = fg_edge_index
        self.num_fgs = fg_x.size(0) if fg_x is not None else None
        self.atom2fg_index = atom2fg_index
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        if key == 'fg_edge_index':
            return self.fg_x.size(0)
        elif key == 'atom2fg_index':
            return torch.tensor([[self.num_nodes], [self.fg_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, value)


class PretrainDataset(InMemoryDataset):
    def __init__(self, root='data/ZINC15',
                 mol_filename='zinc15_250k.txt',
                 fg_corpus_filename='fg_corpus.txt',
                 mol2fgs_filename='mol2fgs_list.json',
                 ):
        self.mol_fn = mol_filename
        self.corpus_fn = fg_corpus_filename
        self.mol2fgs_fn = mol2fgs_filename
        super().__init__(root=root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return [self.mol_fn, self.corpus_fn, self.mol2fgs_fn]

    @property
    def processed_file_names(self):
        return osp.splitext(self.raw_file_names[0])[0] + '.pt'

    def get(self, idx):
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        return data

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            smiles_list = f.read().splitlines()
        with open(self.raw_paths[1], 'r') as f:
            fg_corpus = f.read().splitlines()
        with open(self.raw_paths[2], 'r') as f:
            mol2fgs = json.load(f)
        print(f"# mol: {len(smiles_list)}   # corpus: {len(fg_corpus)}")

        data_list = []
        for smiles, fgs in tqdm(zip(smiles_list, mol2fgs)):
            mol = Chem.MolFromSmiles(smiles)
            atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fpvec = np.zeros(0)
            DataStructs.ConvertToNumpyArray(fp, fpvec)
            fgvec = np.zeros(len(fg_corpus))
            idx = []
            for fg in fgs:
                try:
                    idx.append(fg_corpus.index(fg))
                except:
                    pass
            fgvec[idx] = 1
            data = MultiGraphData(x=torch.Tensor(atom_features),
                                  edge_index=torch.LongTensor(bond_list).reshape(-1, 2).transpose(1, 0),
                                  edge_attr=torch.Tensor(bond_features).reshape(-1, BOND_DIM),
                                  fg_x=torch.Tensor(fg_features),
                                  fg_edge_index=torch.LongTensor(fg_edge_list).reshape(-1, 2).transpose(1, 0),
                                  fg_edge_attr=torch.Tensor(fg_edge_features).reshape(-1, FG_EDGE_DIM),
                                  atom2fg_index=torch.LongTensor(atom2fg_list).reshape(-1, 2).transpose(1, 0),
                                  fp=torch.Tensor(fpvec).reshape(1, -1),
                                  fg=torch.Tensor(fgvec).reshape(1, -1))
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MoleculeNetDataset():
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, y, w = self.dataset[index]
        data = MultiGraphData(x=torch.Tensor(atom_features),
                              edge_index=torch.LongTensor(bond_list).reshape(-1, 2).transpose(1, 0),
                              edge_attr=torch.Tensor(bond_features).reshape(-1, BOND_DIM),
                              fg_x=torch.Tensor(fg_features),
                              fg_edge_index=torch.LongTensor(fg_edge_list).reshape(-1, 2).transpose(1, 0),
                              fg_edge_attr=torch.Tensor(fg_edge_features).reshape(-1, FG_EDGE_DIM),
                              atom2fg_index=torch.LongTensor(atom2fg_list).reshape(-1, 2).transpose(1, 0),
                              y=torch.Tensor(y),
                              w=torch.Tensor(w))
        return data

    def __len__(self):
        return len(self.dataset)


class DDIDataset(InMemoryDataset):
    def __init__(self, root='data/DDI/ZhangDDI',
                 drug_filename='drug_list_zhang.csv',
                 ddi_filename='ZhangDDI_train.csv'):
        self.drug_fn = drug_filename
        self.ddi_fn = ddi_filename
        super().__init__(root=root)

        self.drugs = torch.load(self.processed_paths[0])
        df = pd.read_csv(os.path.join(self.root, self.ddi_fn), usecols=['smiles_1', 'smiles_2', 'label'])
        self.ddi = df.values

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.drug_fn

    @property
    def processed_file_names(self):
        return osp.splitext(self.raw_file_names)[0] + '.pt'

    def __getitem__(self, idx):
        id1, id2, label = self.ddi[idx]
        return self.drugs[id1], self.drugs[id2], torch.Tensor([float(label)])

    def __len__(self):
        return len(self.ddi)

    def process(self):
        df = pd.read_csv(self.raw_paths[0], usecols=['drugbank_id', 'smiles'])
        print(f"# drugs: {len(df)}")

        data_dict = {}
        for _, drug in tqdm(df.iterrows()):
            id, smiles = drug['drugbank_id'], drug['smiles']
            mol = Chem.MolFromSmiles(smiles)
            atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
            if fg_features == []:  # C
                print(f"{smiles} cannot be converted to FG graph")
                continue
            data = MultiGraphData(x=torch.Tensor(atom_features),
                                  edge_index=torch.LongTensor(bond_list).reshape(-1, 2).transpose(1, 0),
                                  edge_attr=torch.Tensor(bond_features).reshape(-1, BOND_DIM),
                                  fg_x=torch.Tensor(fg_features),
                                  fg_edge_index=torch.LongTensor(fg_edge_list).reshape(-1, 2).transpose(1, 0),
                                  fg_edge_attr=torch.Tensor(fg_edge_features).reshape(-1, FG_EDGE_DIM),
                                  atom2fg_index=torch.LongTensor(atom2fg_list).reshape(-1, 2).transpose(1, 0))
            data_dict[smiles] = data

        torch.save(data_dict, self.processed_paths[0])


if __name__ == '__main__':
    dataset = PretrainDataset(mol_filename='zinc15_250k.txt',
                              fg_corpus_filename='fg_corpus.txt',
                              mol2fgs_filename='mol2fgs_list.json')
    data = dataset[0]