import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from proteinpointnet import get_model


class GATNet(torch.nn.Module):
    def __init__(self, num_feature, out_feature, him):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, him, heads=8, concat=True, dropout=0.2)
        self.GAT2 = GATConv(8 * him, out_feature, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.GAT2(x, edge_index)
        return x


class Graph_Net(torch.nn.Module):
    def __init__(self, hidden=512, feature_fusion=None, class_num=7):
        super(Graph_Net, self).__init__()
        self.ppc = get_model()
        self.gat = GATNet(256, 512, 10)
        self.feature_fusion = feature_fusion
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)

    def forward(self, x, edge_index, train_edge_id, p=0.5):
        x = x[:, :, :16]
        x = x.transpose(2, 1)
        x = self.ppc(x)

        x = self.gat(x, edge_index)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc2(x)

        return x

