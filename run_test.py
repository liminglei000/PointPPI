import os


def run_func(ppi_path, pseq_path, vec_path, point_path, protein_max_length, index_path, gnn_model, test_all):
    os.system("python gnn_test.py \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --point_path={} \
            --protein_max_length={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(ppi_path, pseq_path, vec_path, point_path, protein_max_length, index_path, gnn_model, test_all))


if __name__ == "__main__":
    task = "SHS27k"
    # task = "SHS148k"

    split_mode = "random_0.2"
    # split_mode = "bfs_0.2"
    # split_mode = "dfs_0.2"

    test_all = "True"
    # test_all = "False"

    gnn_model = "save_model/SHS27k_True1_" + split_mode + "_False/gnn_model_train.ckpt"

    ppi_path = "dataset/" + task + "/protein.actions." + task + ".STRING.txt"
    pseq_path = "dataset/" + task + "/protein." + task + ".sequences.dictionary.tsv"
    vec_path = "dataset/" + task + "/vec5_CTC.txt"
    point_path = "dataset/" + task + "/protein_point_dim/"
    index_path = "dataset_split/" + task + "_split/" + split_mode + ".txt"
    protein_max_length = 1024

    run_func(ppi_path, pseq_path, vec_path, point_path, protein_max_length, index_path, gnn_model, test_all)
