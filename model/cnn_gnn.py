import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class CNN_GNN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=64,
                 n_filters=32, embed_dim=128, output_dim=64, dropout=0.6):

        super(CNN_GNN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, 3*output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=200, out_channels=128, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*32, 256)
        self.fc2_xt = nn.Linear(256, output_dim)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(32)

        # combined layers
        self.fc1 = nn.Linear(4*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        drugstr = data.drugstr
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        embedded_xt = self.embedding_xt(drugstr)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = self.relu(conv_xt)
        xt = self.conv_xt_2(xt)
        # flatten
        xt = self.globalmaxpool(xt)
        xt = xt.view(-1, 32*32)

        xt = self.fc1_xt(xt)
        xt = self.dropout(xt)
        xt = self.fc2_xt(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out