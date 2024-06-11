import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from sklearn.model_selection import train_test_split

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def smile_to_string(smile):
    x = np.zeros(200)
    for i,ch in enumerate(smile[:200]):
        x[i] = smile_dict[ch]
    return x
# from DeepDTA data

smile_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

all_prots = []
data = pd.read_csv("bace.csv")

np.random.seed(0)
X = data.mol
Y1 = data.iloc[:,8]
Y2 = data.iloc[:,3]
Y = pd.concat([Y1,Y2], axis=1)
X_train, X_test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.1)
X1 = X_train.values.tolist()
X2 = X_test.values.tolist()
Y1 = Y_Train.iloc[:,0].values.tolist()
Y1c = Y_Train.iloc[:,1].values.tolist()
Y2 = Y_Test.iloc[:,0].values.tolist()
Y2c = Y_Test.iloc[:,1].values.tolist()

with open('data/bace_train.csv', 'w') as f:
    f.write('smiles,pIC50,Class\n')
    for i in range(len(X_train)):
        ls = []
        ls += [X1[i]]
        ls += [Y1[i]]
        ls += [Y1c[i]]
        f.write(','.join(map(str, ls)) + '\n')

with open('data/bace_test.csv', 'w') as f:
    f.write('smiles,pIC50,Class\n')
    for i in range(len(X_test)):
        ls = []
        ls += [X2[i]]
        ls += [Y2[i]]
        ls += [Y2c[i]]
        f.write(','.join(map(str, ls)) + '\n')

smiles = []
opts = ['train', 'test']
for opt in opts:
    df = pd.read_csv('data/bace_' + opt + '.csv')
    smiles += list(df['smiles'])
smiles = set(smiles)
smile_graph = {}
smile_string = {}
for smile in smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
    s = smile_to_string(smile)
    smile_string[smile] = s

processed_data_file_train = 'data/processed/bace_train.pt'
processed_data_file_test = 'data/processed/bace_test.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    df = pd.read_csv('data/bace_train.csv')
    train_bace, train_Y = list(df['smiles']), list(df['pIC50'])
    train_drugs, train_Y = np.asarray(train_bace), np.asarray(train_Y)
    df = pd.read_csv('data/bace_test.csv')
    test_bace, test_Y = list(df['smiles']), list(df['pIC50'])
    test_drugs, test_Y = np.asarray(test_bace), np.asarray(test_Y)

    # make data PyTorch Geometric ready
    print('preparing bace_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='bace_train', xd=train_drugs, y=train_Y,
                                smile_graph=smile_graph, smile_string=smile_string)
    print('preparing bace_test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset='bace_test', xd=test_drugs, y=test_Y,
                               smile_graph=smile_graph, smile_string=smile_string)
    print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
else:
    print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')