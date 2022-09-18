from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from scipy.constants import physical_constants

hartree2eV = physical_constants['hartree-electron volt relationship'][0]
DTYPE = np.float32
DTYPE_INT = np.int32


class QM9DGLDataset(Dataset):
    """QM9 dataset."""
    num_bonds = 4
    num_atom_features = 6
    input_keys = ['mol_id', 'num_atoms', 'num_bonds', 'x', 'one_hot',
                  'atomic_numbers', 'edge']

    unit_conversion = {'mu': 1.0,
                       'alpha': 1.0,
                       'homo': hartree2eV,
                       'lumo': hartree2eV,
                       'gap': hartree2eV,
                       'r2': 1.0,
                       'zpve': hartree2eV,
                       'u0': hartree2eV,
                       'u298': hartree2eV,
                       'h298': hartree2eV,
                       'g298': hartree2eV,
                       'cv': 1.0}
    mapping = {1:0,
               6:1,
               7:2,
               8:3,
               9:4}

    def __init__(self, data_path: str, task: str, file_name: str='', mode: str = 'train', 
                 transform=None, fully_connected: bool = False):
        """Create a dataset object
        Args:
            data_path: path to data
            task: target task ["homo", ...]
            mode: [train/valid/test] mode
            transform: data augmentation functions
            fully_connected: return a fully connected graph
        """
        super(QM9DGLDataset, self).__init__()
        assert mode in ['train', 'test']
        self.data_path = data_path
        self.task = task
        self.mode = mode
        self.transform = transform
        self.fully_connected = fully_connected

        # Encode and extra bond type for fully connected graphs
        self.num_bonds += fully_connected

        self.inputs, self.targets = self.load_data(file_name)

        # TODO: use the training stats unlike the other papers
        # self.mean = np.mean(self.targets)
        # self.std = np.std(self.targets)

        print(f"Loaded {mode}-set, task: {task}, source: {self.data_path}, length: {len(self)}")

    def __len__(self):
        return len(self.inputs['mol_id'])

    def train_val_random_split(self, train_ratio=0.8):
        indices = np.arange(self.__len__())
        np.random.shuffle(indices)

        train_size = int(self.__len__() * train_ratio)
        train_indices = indices[:train_size]
        val_indices =  indices[train_size:]
        
        return Subset(self, train_indices), Subset(self, val_indices)

    def load_data(self, file_name: str):
        # Load dict
        file_path = Path(self.data_path) / file_name
        data = torch.load(str(file_path))

        # Filter out the inputs
        inputs = {key: data[key] for key in self.input_keys}

        # Filter out the targets and population stats
        if self.mode == "train":
            targets = data[self.task].astype(DTYPE)
        else:
            targets = None

        return inputs, targets

    # def get_target(self, idx, normalize=True):
    #     target = self.targets[idx]
    #     if normalize:
    #         target = (target - self.mean) / self.std
    #     return target

    # def norm2units(self, x, denormalize=True, center=True):
    #     # Convert from normalized to QM9 representation
    #     if denormalize:
    #         x = x * self.std
    #         # Add the mean: not necessary for error computations
    #         if not center:
    #             x += self.mean
    #     x = self.unit_conversion[self.task] * x
    #     return x

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)), data] = 1
        return one_hot

    def _get_adjacency(self, n_atoms):
        # Adjust adjacency structure
        seq = np.arange(n_atoms)
        src = seq[:, None] * np.ones((1, n_atoms), dtype=np.int32)
        dst = src.T
        ## Remove diagonals and reshape
        src[seq, seq] = -1
        dst[seq, seq] = -1
        src, dst = src.reshape(-1), dst.reshape(-1)
        src, dst = src[src > -1], dst[dst > -1]

        return src, dst

    def get(self, key, idx):
        return self.inputs[key][idx]

    def connect_fully(self, edges, num_atoms):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        adjacency = {}
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    adjacency[(i, j)] = self.num_bonds - 1

        # Add bonded edges
        for idx in range(edges.shape[0]):
            adjacency[(edges[idx, 0], edges[idx, 1])] = edges[idx, 2]
            adjacency[(edges[idx, 1], edges[idx, 0])] = edges[idx, 2]

        # Convert to numpy arrays
        src = []
        dst = []
        w = []
        for edge, weight in adjacency.items():
            src.append(edge[0])
            dst.append(edge[1])
            w.append(weight)

        return np.array(src), np.array(dst), np.array(w)

    @staticmethod
    def connect_partially(edge):
        src = np.concatenate([edge[:, 0], edge[:, 1]])
        dst = np.concatenate([edge[:, 1], edge[:, 0]])
        w = np.concatenate([edge[:, 2], edge[:, 2]])
        return src, dst, w

    def __getitem__(self, idx):
        # Load node features
        num_atoms = self.get('num_atoms', idx)
        x = self.get('x', idx)[:num_atoms].astype(DTYPE)
        one_hot = self.get('one_hot', idx)[:num_atoms].astype(DTYPE)
        atomic_numbers = self.get('atomic_numbers', idx)[:num_atoms].astype(DTYPE_INT).flatten()
        atomic_numbers = np.vectorize(self.mapping.get)(atomic_numbers)
        mol_id = self.get('mol_id', idx)

        # Load edge features
        num_bonds = self.get('num_bonds', idx)
        edge = self.get('edge', idx)[:num_bonds]
        edge = np.asarray(edge, dtype=DTYPE_INT)

        # Augmentation on the coordinates
        if self.transform:
            x = self.transform(x).astype(DTYPE)

        # Create nodes
        if self.fully_connected:
            src, dst, w = self.connect_fully(edge, num_atoms)
        else:
            src, dst, w = self.connect_partially(edge)
        w = self.to_one_hot(w, self.num_bonds).astype(DTYPE)

        # Create graph
        G = dgl.graph((src, dst))

        # Add node features to graph
        G.ndata['x'] = torch.tensor(x)  # [num_atoms,3]
        G.ndata['f'] = torch.tensor(atomic_numbers)  # [num_atoms,1]

        # Add edge features to graph
        G.edata['f'] = torch.tensor(w)  # [num_atoms,4] - Use edge weights as primary edge features

        # y = self.get_target(idx, normalize=True).astype(DTYPE)
        # y = np.array([y])
        if self.mode == "train":
            y = self.targets[idx]
            return G, y
        else:
            return G

    def collate_fn(self, samples):
        if self.mode == "train":
            graphs, y = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(y).view(-1, 1)
        else:
            graphs = list(samples)
            batched_graph = dgl.batch(graphs)
            return batched_graph


class TargetNormalizer:
    def __init__(self, train_indices: np.ndarray, targets: np.ndarray):
        train_targets = targets[train_indices]
        self.mean = np.mean(train_targets)
        self.std = np.std(train_targets)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return (x * self.std) + self.mean