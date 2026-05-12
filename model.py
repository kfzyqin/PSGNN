import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.nn import init
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder

# # PSGNN layer, only pick closest node for message passing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree

from gcn_lib.sparse import GENConv, norm_layer
from utils import get_anchor_dist
from utils_nn import PositionalEncoding
import numpy as np


####################### Basic Ops #############################


class PSGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, to_use_trans=False, to_use_pos=False):
        super(PSGNN_layer, self).__init__()
        self.input_dim = input_dim
        # self.dist_trainable = dist_trainable
        self.dist_trainable = False

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        # Transformer
        self.to_use_trans = to_use_trans
        self.to_use_pos = to_use_pos
        # nhead = 4
        nhead = 2
        # nlayers = 3
        # nlayers = 2
        nlayers = 1
        trans_dropout = 0.5
        self.pos_encoder = PositionalEncoding(output_dim, max_len=30000)
        encoder_layers = TransformerEncoderLayer(output_dim, nhead, dropout=trans_dropout)
        self.trans_enc = TransformerEncoder(encoder_layers, nlayers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))

        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d

        # To use positions
        if self.to_use_pos:
            messages = self.pos_encoder(messages, dists_argmax[0])

        # Here, we add a transformer and different positions for different anchors.
        if self.to_use_trans:
            messages = messages.view(
                messages.shape[1], messages.shape[0], messages.shape[2])
            messages = self.trans_enc(messages)
            messages = messages.view(
                messages.shape[1], messages.shape[0], messages.shape[2])

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


####################### NNs #############################




class ResGCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(ResGCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = GENConv(feature_dim, hidden_dim)
        else:
            self.conv_first = GENConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([GENConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = GENConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)

        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)

        # We make x features all the same
        # x = torch.ones_like(x).to(x.device)

        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x




class PSGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PSGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout

        self.to_use_trans = kwargs['args'].to_use_trans
        self.to_use_pos = kwargs['args'].to_use_pos
        self.anchor_arg = kwargs['args'].anchor_num

        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PSGNN_layer(feature_dim, hidden_dim, to_use_trans=self.to_use_trans,
                                         to_use_pos=self.to_use_pos)
        else:
            self.conv_first = PSGNN_layer(input_dim, hidden_dim, to_use_trans=self.to_use_trans,
                                         to_use_pos=self.to_use_pos)
        if layer_num > 1:
            self.conv_hidden = nn.ModuleList([PSGNN_layer(hidden_dim, hidden_dim, to_use_trans=self.to_use_trans,
                                                         to_use_pos=self.to_use_pos) for i in range(layer_num - 2)])
            self.conv_out = PSGNN_layer(hidden_dim, output_dim, to_use_trans=self.to_use_trans,
                                       to_use_pos=self.to_use_pos)

        # anchor select GCN
        self.anchor_selection_m = kwargs['args'].anchor_selection_m
        if self.anchor_selection_m == 'learning':
            self.anchor_gcn = ResGCN(input_dim, feature_dim, hidden_dim, output_dim,
                                     feature_pre=True, layer_num=3, dropout=True)
            self.acr_gcn_linear = nn.Linear(output_dim, 1)

        self.data_dict = {}

        self.protect_time = 10
        self.protect_count = 0

    def forward(self, data):
        x = data.x

        self.anchor_num = int(1 * int(np.log2(len(x))))

        if self.feature_pre:
            x = self.linear_pre(x)

        if self.anchor_selection_m == 'learning':
            anchor_gcn_out = self.anchor_gcn(data)
            anchor_gcn_out = self.acr_gcn_linear(anchor_gcn_out)
            anchor_gcn_out = F.normalize(anchor_gcn_out, p=2, dim=0)
            dists, dist_argsort = get_anchor_dist(anchor_gcn_out, data.dists, anchor_num=self.anchor_num)
            data.dists_max = dists
            data.dists_argmax = dist_argsort

        elif self.anchor_selection_m == 'max_degree':
            node_degree = degree(data.edge_index[0], num_nodes=len(x))
            if str(data) not in self.data_dict:
                dists, dist_argsort = get_anchor_dist(node_degree, data.dists, anchor_num=self.anchor_num)
                self.data_dict[str(data)] = dists, dist_argsort
            else:
                dists, dist_argsort = self.data_dict[str(data)]
            data.dists_max = dists
            data.dists_argmax = dist_argsort

        elif self.anchor_selection_m == 'random':
            rand_tensor = torch.rand((len(x),)).to(x.device)
            if str(data) not in self.data_dict:
                dists, dist_argsort = get_anchor_dist(rand_tensor, data.dists, anchor_num=self.anchor_num)
                self.data_dict[str(data)] = dists, dist_argsort
            else:
                dists, dist_argsort = self.data_dict[str(data)]
            data.dists_max = dists
            data.dists_argmax = dist_argsort

        elif self.anchor_selection_m == 'betweenness' or \
                self.anchor_selection_m == 'harmonic' or \
                self.anchor_selection_m == 'closeness' or \
                self.anchor_selection_m == 'load':

            if str(data) not in self.data_dict:
                data_nx = tg.utils.to_networkx(data)
                if self.anchor_selection_m == 'betweenness':
                    cen_nodes = nx.algorithms.centrality.betweenness_centrality(data_nx)
                elif self.anchor_selection_m == 'harmonic':
                    cen_nodes = nx.algorithms.centrality.harmonic_centrality(data_nx)
                elif self.anchor_selection_m == 'closeness':
                    cen_nodes = nx.algorithms.centrality.closeness_centrality(data_nx)
                elif self.anchor_selection_m == 'load':
                    cen_nodes = nx.algorithms.centrality.load_centrality(data_nx)
                else:
                    raise NotImplementedError('Not implemented centrality. ')
                central_list = []
                for i in range(len(cen_nodes)):
                    central_list.append(cen_nodes[i])
                central_list = torch.tensor(central_list).to(x.device)

                dists, dist_argsort = get_anchor_dist(central_list, data.dists, anchor_num=self.anchor_num)
                self.data_dict[str(data)] = dists, dist_argsort
            else:
                dists, dist_argsort = self.data_dict[str(data)]

            data.dists_max = dists
            data.dists_argmax = dist_argsort

        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position








