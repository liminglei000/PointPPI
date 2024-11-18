import os
import time
import math
import random
import numpy as np
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, save_checkpoint
import mindspore.ops as ops
from mindspore import dtype as mstype
import mindspore.ops.functional as F
# 如果需要，可以使用MindSpore的SummaryRecord来替代SummaryWriter
from mindspore.train.summary import SummaryRecord

from gnn_data import GNN_DATA
from gnn_model import Graph_Net
from utils import Metrictor_PPI, print_file

from parameter_setting import *

np.random.seed(1)
ms.set_seed(1)


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
                    help='train ppi graph construct by train or all(train with test)')
parser.add_argument('--batch_size', default=batch_size, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=epochs, type=int,
                    help='train epoch number')


def train(model, graph, ppi_list, loss_fn, optimizer,
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

        model.set_train(True)

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)
        
        
        for step in range(steps):
            if step == steps - 1:
                train_edge_id = graph.train_mask_got[step * batch_size:] if got else graph.train_mask[step * batch_size:]
            else:
                train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size] if got else graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output = model(graph.x, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output = model(graph.x, graph.edge_index, train_edge_id)
                label = graph.edge_attr_1[train_edge_id]

            label = Tensor(label, mstype.float32)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).astype(mstype.float32)

            metrics = Metrictor_PPI(pre_result.asnumpy(), label.asnumpy())

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.asnumpy()

            summary_writer.add_value('train_loss', loss.asnumpy(), global_step)
            summary_writer.add_value('train_precision', metrics.Precision, global_step)
            summary_writer.add_value('train_recall', metrics.Recall, global_step)
            summary_writer.add_value('train_F1', metrics.F1, global_step)
            summary_writer.add_value('train_auc', metrics.auc, global_step)
            summary_writer.add_value('train_hmloss', metrics.hmloss, global_step)

            global_step += 1

        save_checkpoint(model, os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.set_train(False)

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)
        
        F.stop_gradient(model)
        for step in range(valid_steps):
            if step == valid_steps - 1:
                valid_edge_id = graph.val_mask[step * batch_size:]
            else:
                valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

            output = model(graph.x, graph.edge_index, valid_edge_id)
            label = graph.edge_attr_1[valid_edge_id]
            label = Tensor(label, mstype.float32)
            loss = loss_fn(output, label)

            valid_loss_sum += loss.asnumpy()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).astype(mstype.float32)

            valid_pre_result_list.append(pre_result.asnumpy())
            valid_label_list.append(label.asnumpy())

        valid_pre_result_list = np.concatenate(valid_pre_result_list, axis=0)
        valid_label_list = np.concatenate(valid_label_list, axis=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss_avg = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler is not None:
            scheduler.step(loss_avg)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            save_checkpoint(model, os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_value('valid_precision', metrics.Precision, global_step)
        summary_writer.add_value('valid_recall', metrics.Recall, global_step)
        summary_writer.add_value('valid_F1', metrics.F1, global_step)
        summary_writer.add_value('valid_loss', valid_loss, global_step)
        summary_writer.add_value('valid_auc', metrics.auc, global_step)
        summary_writer.add_value('valid_hmloss', metrics.hmloss, global_step)

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, auc: {}, hmloss: {}, Best valid_f1: {}, in {} epoch"
            .format(epoch, loss_avg, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                    metrics.auc, metrics.hmloss, global_best_valid_f1, global_best_valid_f1_epoch),
            save_file_path=result_file_path)


def main():
    args = parser.parse_args()

    # 设置MindSpore上下文
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path)

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

    graph.edge_index_got = ops.Concat(axis=1)((ppi_data.edge_index[:, graph.train_mask],
                                               ppi_data.edge_index[:, graph.train_mask][[1, 0]]))
    graph.edge_attr_got = ops.Concat(axis=0)((ppi_data.edge_attr[graph.train_mask], ppi_data.edge_attr[graph.train_mask]))
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    model = Graph_Net().astype(mstype.float32)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = nn.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    loss_fn = nn.BCEWithLogitsLoss()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    time_stamp = time.strftime("%m-%d-%H-%M-%S")
    save_path = os.path.join(args.save_path, "{}_{}".format(args.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        args_dict = vars(args)
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('max_length = {}'.format(Protein_Max_Length))
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    # 使用MindSpore的SummaryRecord替代SummaryWriter
    summary_writer = SummaryRecord(save_path)

    train(model, graph, ppi_list, loss_fn, optimizer,
          result_file_path, summary_writer, save_path,
          batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler,
          got=args.graph_only_train)

    summary_writer.close()


if __name__ == "__main__":
    main()
