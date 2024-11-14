import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import composite as C
from mindspore.common.initializer import Normal
from mindspore import Tensor

class SAGENet(nn.Cell):
    def __init__(self, num_features, out_features, hidden_dim):
        super(SAGENet, self).__init__()
        self.SAGE1 = ops.SAGEConv(num_features, hidden_dim)
        self.SAGE2 = ops.SAGEConv(hidden_dim, out_features)

    def construct(self, x, edge_index):
        x = self.SAGE1(x, edge_index)
        x = ops.Dropout(keep_prob=0.9)(ops.ReLU()(x))
        x = self.SAGE2(x, edge_index)
        return x

class GINNet(nn.Cell):
    def __init__(self, num_features, out_features, hidden_dim, num_layers):
        super(GINNet, self).__init__()
        self.convs = nn.CellList()
        self.convs.append(ops.GINConv(nn.SequentialCell(
            nn.Dense(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU()
        )))

        for _ in range(num_layers - 1):
            self.convs.append(ops.GINConv(nn.SequentialCell(
                nn.Dense(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dense(hidden_dim, hidden_dim),
                nn.ReLU()
            )))

        self.lin = nn.Dense(hidden_dim, out_features)

    def construct(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = ops.Dropout(keep_prob=0.9)(ops.ReLU()(x))

        x = self.lin(x)
        return x

class GATNet(nn.Cell):
    def __init__(self, num_feature, out_feature, hidden):
        super(GATNet, self).__init__()
        self.GAT1 = ops.GATConv(num_feature, hidden, num_heads=8, concat=True, dropout_ratio=0.2)
        self.GAT2 = ops.GATConv(8 * hidden, out_feature, dropout_ratio=0.2)

    def construct(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = ops.ReLU()(x)
        x = self.GAT2(x, edge_index)
        return x

class WeightedFeatureFusion(nn.Cell):
    def __init__(self, feature_dim):
        super(WeightedFeatureFusion, self).__init__()
        self.weights = nn.Parameter(Tensor(np.ones(3).astype(np.float32)))

    def construct(self, input1, input2, input3):
        weighted_sum = self.weights[0] * input1 + self.weights[1] * input2 + self.weights[2] * input3
        output = ops.BatchNorm()(weighted_sum)
        return output

class GraphNet(nn.Cell):
    def __init__(self, hidden=512, feature_fusion=None, class_num=7):
        super(GraphNet, self).__init__()

        self.gcn = SAGENet(256, 512, 128)
        self.gin = GINNet(256, 512, 128, 2)
        self.gat = GATNet(256, 512, 10)
        self.fusion_model = WeightedFeatureFusion(512)

        self.feature_fusion = feature_fusion
        self.lin1 = nn.Dense(hidden, hidden)
        self.lin2 = nn.Dense(hidden, hidden)
        self.fc2 = nn.Dense(hidden, class_num)

        # Assume get_model() returns a MindSpore model
        self.ppc = get_model()

    def construct(self, x, edge_index, train_edge_id, p=0.5):
        x = x[:, :, :16]
        x = x.transpose(2, 1)
        x = self.ppc(x)

        x = self.fusion_model(self.gcn(x, edge_index), self.gin(x, edge_index), self.gat(x, edge_index))

        x = ops.ReLU()(self.lin1(x))
        x = ops.Dropout(keep_prob=p)(x)
        x = self.lin2(x)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = ops.Concat(1)(x1, x2)
        else:
            x = ops.Mul()(x1, x2)
        x = self.fc2(x)

        return x
