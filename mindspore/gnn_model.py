import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore import Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore_gl.nn import GINConv, SAGEConv, GATConv

from proteinpointnet import get_model

class SAGENet(Cell):
    def __init__(self, num_features, out_features, hidden_dim):
        super(SAGENet, self).__init__()
        self.SAGE1 = SAGEConv(num_features, hidden_dim)
        self.SAGE2 = SAGEConv(hidden_dim, out_features)
        self.dropout = ops.dropout(p=0.1, training=True)
        self.relu = nn.ReLU()

    def construct(self, x, edge_index):
        x = self.SAGE1(x, edge_index)
        x = self.dropout(self.relu(x))
        x = self.SAGE2(x, edge_index)
        return x

class GINNet(Cell):
    def __init__(self, num_features, out_features, hidden_dim, num_layers):
        super(GINNet, self).__init__()
        self.convs = nn.CellList()
        self.convs.append(GINConv(nn.SequentialCell([
            nn.Dense(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU()
        ])))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.SequentialCell([
                nn.Dense(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dense(hidden_dim, hidden_dim),
                nn.ReLU()
            ])))

        self.lin = nn.Dense(hidden_dim, out_features)
        self.dropout = ops.dropout(p=0.1, training=True)
        self.relu = nn.ReLU()

    def construct(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(self.relu(x))
        x = self.lin(x)
        return x

class GATNet(Cell):
    def __init__(self, num_feature, out_feature, him):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, him, num_heads=8, concat=True, dropout_rate=0.2)
        self.GAT2 = GATConv(8 * him, out_feature, dropout_rate=0.2)
        self.relu = nn.ReLU()

    def construct(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = self.relu(x)
        x = self.GAT2(x, edge_index)
        return x

class WeightedFeatureFusion(Cell):
    def __init__(self, feature_dim):
        super(WeightedFeatureFusion, self).__init__()
        self.weights = Parameter(Tensor(mnp.ones(3), mindspore.float32))
        self.batch_norm = ops.batch_norm(running_mean=None, running_var=None, training=True, weight=None, bias=None)

    def construct(self, input1, input2, input3):
        weighted_sum = self.weights[0] * input1 + self.weights[1] * input2 + self.weights[2] * input3
        output = self.batch_norm(weighted_sum)
        return output

class Graph_Net(Cell):
    def __init__(self, hidden=512, feature_fusion=None, class_num=7):
        super(Graph_Net, self).__init__()
        
        self.gcn = SAGENet(256, 512, 128)
        self.gin = GINNet(256, 512, 128, 2)
        self.gat = GATNet(256, 512, 10)
        self.fusion_model = WeightedFeatureFusion(512)
        
        self.feature_fusion = feature_fusion
        self.lin1 = nn.Dense(hidden, hidden)
        self.lin2 = nn.Dense(hidden, hidden)
        self.fc2 = nn.Dense(hidden, class_num)
        
        self.ppc = get_model()
        
        self.dropout = ops.dropout()
        self.relu = nn.ReLU()
        self.concat = ops.Concat(axis=1)
        self.mul = ops.mul()

    def construct(self, x, edge_index, train_edge_id, p=0.5):
        x = x[:, :, :16]
        x = ops.Transpose()(x, (0, 2, 1))
        x = self.ppc(x)

        x_gcn = self.gcn(x, edge_index)
        x_gin = self.gin(x, edge_index)
        x_gat = self.gat(x, edge_index)

        x = self.fusion_model(x_gcn, x_gin, x_gat)

        x = self.relu(self.lin1(x))
        x = self.dropout(x, p=p, training=self.training)
        x = self.lin2(x)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = self.concat((x1, x2))
        else:
            x = self.mul(x1, x2)
        x = self.fc2(x)
        return x
