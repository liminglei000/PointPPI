import torch
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# task = "SHS27k"
task = "SHS148k"

# split_mode = "random"
# split_mode = "bfs"
split_mode = "dfs"

split_new = "False"
# split_new = "True"

graph_only_train = "False"
# graph_only_train = "True"
description = task + "_" + split_mode
# description = task + "_" + "GCT" + "_" + split_mode

batch_size = 2048
epochs = 500
learning_rate = 0.001
protein_max_length = 1024

ppi_path = "dataset/" + task + "/protein.actions." + task + ".STRING.txt"
pseq_path = "dataset/" + task + "/protein." + task + ".sequences.dictionary.tsv"
train_valid_index_path = "dataset_split/" + task + "_split/" + split_mode + ".txt"
use_lr_scheduler = "True"
save_path = "save_model/"
vec_path = "dataset/" + task + "/vec5_CTC.txt"
point_path = "dataset/" + task + "/protein_point_dim/"

similarity_matrix_path = "dataset/" + task + "/protein_similarity_" + task + ".txt"
alpha = 0.01
# similarity_matrix = np.loadtxt(similarity_matrix_path)
similarity_matrix = np.zeros((2, 2))
use_similarity = False
