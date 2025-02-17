import math
import json
import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from gnn_data import GNN_DATA
from gnn_model import Graph_Net
from utils import Metrictor_PPI


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def set_args():
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--ppi_path', default=None, type=str, help="ppi path")
    parser.add_argument('--pseq_path', default=None, type=str, help="protein sequence path")
    parser.add_argument('--vec_path', default=None, type=str, help="protein sequence path")
    parser.add_argument('--point_path', default=None, type=str, help="protein point path")
    parser.add_argument('--protein_max_length', default=None, type=int, help="protein max length")
    parser.add_argument('--index_path', default=None, type=str, help='cnn_rnn and gnn unified train and valid ppi index')
    parser.add_argument('--gnn_model', default=None, type=str, help="gnn trained model")
    parser.add_argument('--test_all', default='False', type=boolean_string, help="test all or test separately")
    return parser.parse_args()


def test(model, graph, test_mask, device, gnn_model="./"):
    valid_predict_list = []
    valid_pre_result_list = []
    valid_label_list = []

    model.eval()
    batch_size = 2048
    valid_steps = math.ceil(len(test_mask) / batch_size)

    for step in tqdm(range(valid_steps)):
        if step == valid_steps-1:
            valid_edge_id = test_mask[step*batch_size:]
        else:
            valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

        output, _ = model(graph.x, graph.edge_index, valid_edge_id)
        label = graph.edge_attr[valid_edge_id]
        label = label.type(torch.FloatTensor).to(device)

        m = nn.Sigmoid()
        pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

        valid_predict_list.append(m(output).cpu().data)
        valid_pre_result_list.append(pre_result.cpu().data)
        valid_label_list.append(label.cpu().data)

    valid_predict_list = torch.cat(valid_predict_list, dim=0)
    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)
    metrics.show_result()
    print("F1:{:.4f}, Auc:{:.4f}, Aupr:{:.4f}, hmloss:{:.4f}".format(metrics.F1, metrics.auc, metrics.aupr, metrics.hmloss))

    np.savetxt(gnn_model+"predict.txt", valid_predict_list, fmt="%.4f", delimiter=",")
    np.savetxt(gnn_model+"label.txt", valid_label_list, fmt="%d", delimiter=",")


def main():
    args = set_args()
    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path, point_path=args.point_path, protein_max_length=args.protein_max_length)
    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []
    for edge in temp: ppi_list.append(list(edge))

    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']
    graph.val_mask = index_dict['valid_index']
    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0

    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Graph_Net().to(device)
    model.load_state_dict(torch.load(args.gnn_model)['state_dict'])
    graph.to(device)

    if args.test_all:
        print("---------------- valid-test-all result --------------------")
        test(model, graph, graph.val_mask, device, args.gnn_model[:-20])
    else:
        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            test(model, graph, graph.test1_mask, device)
        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            test(model, graph, graph.test2_mask, device)
        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            test(model, graph, graph.test3_mask, device)

if __name__ == "__main__":
    main()
