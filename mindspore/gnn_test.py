import os
import time
import math
import json
import random
import numpy as np
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net

from tqdm import tqdm

from gnn_data import GNN_DATA
from gnn_model import Graph_Net
from utils import Metrictor_PPI, print_file

from parameter_setting import *

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--gnn_model', default=None, type=str,
                    help="gnn trained model")
parser.add_argument('--test_all', default='False', type=boolean_string,
                    help="test all or test separately")

def test(model, graph, test_mask, device=None):
    valid_pre_result_list = []
    valid_label_list = []

    model.set_train(False)  # 设置模型为评估模式

    batch_size = 256

    valid_steps = math.ceil(len(test_mask) / batch_size)

    for step in tqdm(range(valid_steps)):
        if step == valid_steps-1:
            valid_edge_id = test_mask[step*batch_size:]
        else:
            valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

        output = model(graph.x, graph.edge_index, valid_edge_id)
        label = graph.edge_attr_1[valid_edge_id]
        label = ms.Tensor(label, ms.float32)
        m = nn.Sigmoid()
        pre_result = (m(output) > 0.5).astype(ms.float32)

        valid_pre_result_list.append(pre_result.asnumpy())
        valid_label_list.append(label.asnumpy())

    valid_pre_result_list = np.concatenate(valid_pre_result_list, axis=0)
    valid_label_list = np.concatenate(valid_label_list, axis=0)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

    metrics.show_result()

    print("Recall: {}, Precision: {}, F1: {}, auc: {}, hmloss: {}".format(
        metrics.Recall, metrics.Precision, metrics.F1, metrics.auc, metrics.hmloss))

def main():

    args = parser.parse_args()

    # 设置MindSpore上下文
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    ppi_data.get_feature_origin(pseq_path=args.pseq_path)

    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).asnumpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    fake_edge_num = 0

    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()

    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict:
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict:
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict:
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict:
            node_vision_dict[ppi[1]] = 0

    vision_num = sum(value == 1 for value in node_vision_dict.values())
    unvision_num = sum(value == 0 for value in node_vision_dict.values())
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:
            test1_mask.append(index)
        elif temp == 1:
            test2_mask.append(index)
        elif temp == 0:
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask), len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    model = Graph_Net()

    # 加载模型参数
    param_dict = load_checkpoint(args.gnn_model)
    load_param_into_net(model, param_dict)

    if args.test_all:
        print("---------------- valid-test-all result --------------------")
        test(model, graph, graph.val_mask)
    else:
        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            test(model, graph, graph.test1_mask)
        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            test(model, graph, graph.test2_mask)
        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            test(model, graph, graph.test3_mask)

if __name__ == "__main__":
    main()
