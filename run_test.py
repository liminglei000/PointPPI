import os


def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, gnn_model, test_all):
    os.system("python gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, test_all))

if __name__ == "__main__":
    description = "test"

    task = 'SHS27K'
    # task = 'STRING'

    mode = 'random'
    # mode = 'bfs'
    # mode = 'dfs'

    ppi_path = "../dataset/" + task + "/protein.actions." + task + ".STRING.txt"
    pseq_path = "../dataset/" + task + "/protein." + task + ".sequences.dictionary.tsv"
    vec_path = "../dataset/" + task + "/vec5_CTC.txt"

    index_path = "../dataset_split/" + task + "_split/" + mode + ".txt"
    gnn_model = "save_model/SHS27k_random_25-08-43-08-43/gnn_model_train.ckpt"

    test_all = "True"
    # test_all = "False"

    run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all)
