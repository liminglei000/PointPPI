import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, GATConv

from proteinpointnet import get_model


class SAGENet(torch.nn.Module):
    def __init__(self, num_features, out_features, hidden_dim):
        super(SAGENet, self).__init__()
        self.SAGE1 = SAGEConv(num_features, hidden_dim)
        self.SAGE2 = SAGEConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = self.SAGE1(x, edge_index)
        x = F.dropout(F.relu(x), p=0.1, training=True)
        x = self.SAGE2(x, edge_index)
        return x


class GINNet(nn.Module):
    def __init__(self, num_features, out_features, hidden_dim, num_layers):
        super(GINNet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )))

        self.lin = nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), p=0.1, training=True)

        x = self.lin(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self, num_feature, out_feature, him):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, him, heads=8, concat=True, dropout=0.2)  # 0.6->0.2
        self.GAT2 = GATConv(8 * him, out_feature, dropout=0.2)  # 0.6->0.2

    def forward(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.GAT2(x, edge_index)
        return x


class WeightedFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super(WeightedFeatureFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, input1, input2, input3):
        weighted_sum = self.weights[0] * input1 + self.weights[1] * input2 + self.weights[2] * input3
        output = F.batch_norm(weighted_sum, running_mean=None, running_var=None, training=True)
        return output


class Graph_Net(torch.nn.Module):
    def __init__(self, hidden=512, feature_fusion=None, class_num=7):
        super(Graph_Net, self).__init__()

        self.gcn = SAGENet(256, 512, 128)
        self.gin = GINNet(256, 512, 128, 2)
        self.gat = GATNet(256, 512, 10)
        self.fusion_model = WeightedFeatureFusion(512)

        self.feature_fusion = feature_fusion
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)

        self.ppc = get_model()

    def forward(self, x, edge_index, train_edge_id, p=0.5):
        x = x[:, :, :16]
        x = x.transpose(2, 1)
        x = self.ppc(x)

        x = self.fusion_model(self.gcn(x, edge_index), self.gin(x, edge_index), self.gat(x, edge_index))

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
