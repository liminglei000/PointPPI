import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from gnn_data import GNN_DATA
from gnn_model import Graph_Net
from utils import Metrictor_PPI, print_file

from tensorboardX import SummaryWriter

from parameter_setting import *

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def process_similarity(x1, x2, dim=1, small_threshold=1e-2):
    similarity_now = F.cosine_similarity(x1, x2, dim=1)
    norm_x1 = torch.norm(x1, dim=dim)
    norm_x2 = torch.norm(x2, dim=dim)
    small_mask_x1 = norm_x1 < small_threshold
    small_mask_x2 = norm_x2 < small_threshold
    similarity_now[small_mask_x1 | small_mask_x2] = 0
    return similarity_now


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default=description, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=ppi_path, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=pseq_path, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=vec_path, type=str,
                    help="protein sequence path")
parser.add_argument('--point_path', default=point_path, type=str,
                    help="protein point path")
parser.add_argument('--protein_max_length', default=protein_max_length, type=int,
                    help="protein max length")
parser.add_argument('--split_new', default=split_new, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default=split_mode, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default=train_valid_index_path, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=use_lr_scheduler, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default=save_path, type=str,
                    help='model save path')
parser.add_argument('--graph_only_train', default=graph_only_train, type=boolean_string,
                    help='train ppi graph conctruct by train or all(train with test)')
parser.add_argument('--batch_size', default=batch_size, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=epochs, type=int,
                    help='train epoch number')


def train(model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=512, epochs=1000, scheduler=None,
          got=False):
    global_step = 0
    global_best_valid_f1 = 0
    global_best_valid_f1_epoch = 0

    truth_edge_num = graph.edge_index.shape[1] // 2

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)
        # print (steps)

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
                output, x1, x2, a, b = model(graph.x, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output, x1, x2, a, b = model(graph.x, graph.edge_index, train_edge_id)
                label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)
            if use_similarity:
                if step == steps - 1:
                    similarity_origin = np.zeros(len(graph.train_mask) - step * batch_size)
                    for i in range(len(graph.train_mask) - step * batch_size):
                        similarity_origin[i] = similarity_matrix[a[i], b[i]]
                else:
                    similarity_origin = np.zeros(batch_size)
                    for i in range(batch_size):
                        similarity_origin[i] = similarity_matrix[a[i], b[i]]
                similarity_origin = torch.tensor(similarity_origin).to(device)

                similarity_now = process_similarity(x1, x2)
                similarity_now = torch.squeeze((similarity_now + 1) / 2)
                loss_mse = ((similarity_now - similarity_origin)[similarity_origin >= 0.5] ** 2).mean()

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

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            # summary_writer.add_scalar('train/loss', loss.item(), global_step)
            # summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            # summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            # summary_writer.add_scalar('train/F1', metrics.F1, global_step)
            # summary_writer.add_scalar('train/auc', metrics.auc, global_step)
            # summary_writer.add_scalar('train/hmloss', metrics.hmloss, global_step)
            #
            # global_step += 1
            # print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}, auc: {}, hmloss: {}"
            #             .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1, metrics.auc, metrics.hmloss))

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

                output, x1, x2, a, b = model(graph.x, graph.edge_index, valid_edge_id)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                if use_similarity:
                    if step == steps - 1:
                        similarity_origin = np.zeros(len(graph.train_mask) - step * batch_size)
                        for i in range(len(graph.train_mask) - step * batch_size):
                            similarity_origin[i] = similarity_matrix[a[i], b[i]]
                    else:
                        similarity_origin = np.zeros(batch_size)
                        for i in range(batch_size):
                            similarity_origin[i] = similarity_matrix[a[i], b[i]]
                    similarity_origin = torch.tensor(similarity_origin).to(device)

                    similarity_now = process_similarity(x1, x2)
                    similarity_now = torch.squeeze((similarity_now + 1) / 2)
                    loss_mse = ((similarity_now - similarity_origin)[similarity_origin >= 0.5] ** 2).mean()

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

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            # print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']), save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)
        summary_writer.add_scalar('valid/auc', metrics.auc, global_step)
        summary_writer.add_scalar('valid/hmloss', metrics.hmloss, global_step)

        print_file("epoch: {}, Training: loss: {}, F1: {}, Valid: loss: {}, F1: {}, Auc:{}, hmloss: {}"
                   .format(epoch, loss, f1, valid_loss, metrics.F1, metrics.auc, metrics.hmloss), save_file_path=result_file_path)


def main():
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path,
                                point_path=args.point_path, protein_max_length=args.protein_max_length)

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new:
        print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data

    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    graph.edge_index_got = torch.cat(
        (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]
    graph.to(device)

    model = Graph_Net().to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    time_stamp = time.strftime("%m-%d-%H-%M-%S")
    save_path = os.path.join(args.save_path, "{}_{}".format(args.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        args_dict = args.__dict__
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('max_length = {}'.format(protein_max_length))
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    summary_writer = SummaryWriter(save_path)

    train(model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler,
          got=args.graph_only_train)

    summary_writer.close()


if __name__ == "__main__":
    main()
