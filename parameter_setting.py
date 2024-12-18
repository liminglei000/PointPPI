import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

split_new = "False"
# split_new = "True"

task = "SHS27k"
# task = "SHS148k"

# use_similarity = False
use_similarity = True
alpha = 1

split_mode = "random_0.2"
# split_mode = "bfs_0.2"
# split_mode = "dfs_0.2"

graph_only_train = "False"
# graph_only_train = "True"

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
similarity_matrix = np.loadtxt(similarity_matrix_path)
protein_protein_similarity_id = []
protein_protein_similarity_val = []
for i in range(similarity_matrix.shape[0]):
    for j in range(i):
        if similarity_matrix[i][j] > 0.5:
            protein_protein_similarity_id.append([j, i])
            protein_protein_similarity_val.append(similarity_matrix[i][j])
