import os
import math
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
from gnn_data import GNN_DATA
from gnn_model import Graph_Net
from utils import Metrictor_PPI, print_file
from parameter_setting import *

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def cal_similarity(x1, x2):
    norm_x1 = torch.norm(x1)
    norm_x2 = torch.norm(x2)
    if norm_x1 < 1e-2 or norm_x2 < 1e-2:
        return torch.tensor(0)
    return F.cosine_similarity(x1, x2, dim=0)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def set_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--task', default=task, type=str, help='task name')
    parser.add_argument('--ppi_path', default=ppi_path, type=str, help="ppi path")
    parser.add_argument('--pseq_path', default=pseq_path, type=str, help="protein sequence path")
    parser.add_argument('--vec_path', default=vec_path, type=str, help="protein sequence path")
    parser.add_argument('--point_path', default=point_path, type=str, help="protein point path")
    parser.add_argument('--protein_max_length', default=protein_max_length, type=int, help="protein max length")
    parser.add_argument('--split_new', default=split_new, type=boolean_string, help='split new index file or not')
    parser.add_argument('--split_mode', default=split_mode, type=str, help='split method, random, bfs or dfs')
    parser.add_argument('--train_valid_index_path', default=train_valid_index_path, type=str, help='train and valid ppi index')
    parser.add_argument('--use_lr_scheduler', default=use_lr_scheduler, type=boolean_string, help="train use learning rate scheduler or not")
    parser.add_argument('--save_path', default=save_path, type=str, help='model save path')
    parser.add_argument('--graph_only_train', default=graph_only_train, type=boolean_string, help='train ppi graph construct by train or all')
    parser.add_argument('--batch_size', default=batch_size, type=int, help="gnn train batch size, edge batch size")
    parser.add_argument('--epochs', default=epochs, type=int, help='train epoch number')
    return parser.parse_args()


def train(model, graph, loss_fn, optimizer, device, result_file_path, save_path, batch_size=512, epochs=1000, scheduler=None, got=False):
    global_best_valid_f1 = 0
    global_best_valid_auc = 0
    global_best_valid_aupr = 0
    global_best_valid_hmloss = 0
    for epoch in range(epochs):
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)
        model.train()
        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output, node_embedding = model(graph.x, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output, node_embedding = model(graph.x, graph.edge_index, train_edge_id)
                label = graph.edge_attr[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            if use_similarity:
                similarity_origin = torch.tensor(protein_protein_similarity_val).to(device)
                similarity_new = []
                for pp in protein_protein_similarity_id:
                    x1 = node_embedding.get(pp[0], None)
                    x2 = node_embedding.get(pp[1], None)
                    if x1 is not None and x2 is not None:
                        similarity_new.append(cal_similarity(x1, x2))
                    else:
                        similarity_new.append(torch.tensor(0))
                similarity_new = (torch.tensor(similarity_new).to(device) + 1) / 2
                loss_mse = ((similarity_new - similarity_origin) ** 2).mean()

                loss = loss_fn(output, label) + alpha * loss_mse
            else:
                loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)
            metrics.show_result()
            f1_sum += metrics.F1
            loss_sum += loss.item()

        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()
        valid_steps = math.ceil(len(graph.val_mask) / batch_size)
        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output, node_embedding = model(graph.x, graph.edge_index, valid_edge_id)
                label = graph.edge_attr[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                if use_similarity:
                    similarity_origin = torch.tensor(protein_protein_similarity_val).to(device)
                    similarity_new = []
                    for pp in protein_protein_similarity_id:
                        x1 = node_embedding.get(pp[0], None)
                        x2 = node_embedding.get(pp[1], None)
                        if x1 is not None and x2 is not None:
                            similarity_new.append(cal_similarity(x1, x2))
                        else:
                            similarity_new.append(torch.tensor(0))
                    similarity_new = (torch.tensor(similarity_new).to(device) + 1) / 2
                    loss_mse = ((similarity_new - similarity_origin) ** 2).mean()

                    loss = loss_fn(output, label) + alpha * loss_mse
                else:
                    loss = loss_fn(output, label)

                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

        metrics.show_result()
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None: scheduler.step(loss)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_auc = metrics.auc
            global_best_valid_aupr = metrics.aupr
            global_best_valid_hmloss = metrics.hmloss
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        print_file("epoch:{}, Train loss:{:.4f} -- Valid loss:{:.4f}, F1:{:.4f}, Auc:{:.4f}, Aupr:{:.4f}, hmloss:{:.4f} -- "
                   "best F1:{:.4f}, best Auc:{:.4f}, best Aupr:{:.4f}, best hmloss:{:.4f}"
            .format(epoch, loss, valid_loss, metrics.F1, metrics.auc, metrics.aupr, metrics.hmloss,global_best_valid_f1,
                    global_best_valid_auc, global_best_valid_aupr, global_best_valid_hmloss), save_file_path=result_file_path)


def main():
    args = set_args()
    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path, point_path=args.point_path, protein_max_length=args.protein_max_length)
    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new: print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, test_size=float(args.split_mode.split('_')[1]), random_new=args.split_new, mode=args.split_mode.split('_')[0])
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data
    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']
    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    graph.edge_index_got = torch.cat((graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr[graph.train_mask], graph.edge_attr[graph.train_mask]), dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]
    graph.to(device)

    model = Graph_Net().to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(args.save_path): os.mkdir(args.save_path)
    if not use_similarity: stamp = str(use_similarity) + "_" + args.split_mode + "_" + str(args.graph_only_train)
    else: stamp = str(use_similarity) + str(alpha) + "_" + args.split_mode + "_" + str(args.graph_only_train)
    save_path = os.path.join(args.save_path, "{}_{}".format(args.task, stamp))  # SHS27k_True0.1_random_0.2_False
    result_file_path = os.path.join(save_path, "valid_results.txt")
    os.mkdir(save_path)

    train(model, graph, loss_fn, optimizer, device, result_file_path, save_path,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler, got=args.graph_only_train)


if __name__ == "__main__":
    main()
