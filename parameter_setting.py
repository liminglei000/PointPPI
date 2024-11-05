import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

task = "SHS27k"
# task = "SHS148k"

split_mode = "random"
# split_mode = "bfs"
# split_mode = "dfs"

# split_new = "True"
split_new = "False"

graph_only_train = "False"
# graph_only_train = "True"

batch_size = 1024
epochs = 500
learning_rate = 0.001
Protein_Max_Length = 1024

description = task + "_" + split_mode

ppi_path = "dataset/" + task + "/protein.actions." + task + ".STRING.txt"
pseq_path = "dataset/" + task + "/protein." + task + ".sequences.dictionary.tsv"
train_valid_index_path = "dataset_split/" + task + "_split/" + split_mode + ".txt"
use_lr_scheduler = "True"
save_path = "save_model/"
vec_path = "dataset/" + task + "/vec5_CTC.txt"
point_path = "dataset/" + task + "/protein_point_dim/"
