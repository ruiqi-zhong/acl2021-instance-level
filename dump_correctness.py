import json
import numpy as np
from utils import invert_labellist, load_tsv, obtain_correctness, obtain_prob
import pickle as pkl
import os

model_sizes = ['mini', 'small', 'medium', 'base', 'large']

# target dumping directory
p_dump_dir = 'correctness_p/'
c_dump_dir = 'correctness/'
e_dump_dir = 'ensemble_c/'


for data_name in ['qqp', 'mnli']:
    p_pkl_path = os.path.join(p_dump_dir, data_name + '.pkl')
    c_pkl_path = os.path.join(c_dump_dir, data_name + '.pkl')
    e_pkl_path = os.path.join(e_dump_dir, data_name + '.pkl')

    # loading template
    f_path_template = 'model_data_for_release/' + data_name + '/results/s{s}p{p}f{f}epoch{i}over3.tsv'
    data_path= 'model_data_for_release/' + data_name + '/data.json'

    # dictionary that contains the results
    size2individual_runs_c = {}
    size2individual_runs_p = {}
    size2ensemble_runs_c = {}

    all_predicts = []

    data = json.load(open(data_path, 'r'))
    label_list = data['label_list']
    label_map = invert_labellist(label_list)
    predict_dicts = data['predict']
    all_predicts += predict_dicts

    for model_size in model_sizes:
        prob_4D, c_4D, ensemble_2D = [], [], []
        for pretrain_seed in range(1, 11):
            prob_3D, c_3D, probs = [], [], []
            for finetune_seed in range(1, 6):
                prob_2D, c_2D = [], []
                for i in range(9, 13):
                    f_path = f_path_template.format(p=pretrain_seed, f=finetune_seed, s=model_size, i=i)
                    print(f_path)
                    p = load_tsv(f_path)
                    if i == 12:
                        probs.append(p)
                    prob_2D.append(obtain_prob(p, predict_dicts, label_map).round(decimals=2))
                    c_2D.append(obtain_correctness(p, predict_dicts, label_map))

                prob_3D.append(prob_2D)
                c_3D.append(c_2D)

            probs_marginalized_f = np.mean(probs, axis=0)
            ensemble_2D.append(obtain_correctness(probs_marginalized_f, predict_dicts, label_map))
            prob_4D.append(prob_3D)
            c_4D.append(c_3D)

        size2individual_runs_p[model_size] = np.array(prob_4D)
        size2individual_runs_c[model_size] = np.array(c_4D)
        size2ensemble_runs_c[model_size] = np.array(ensemble_2D)

    for size in model_sizes:
        size2individual_runs_p[size] = size2individual_runs_p[size].transpose((3, 0, 1, 2))
        size2individual_runs_c[size] = size2individual_runs_c[size].transpose((3, 0, 1, 2))
        size2ensemble_runs_c[size] = size2ensemble_runs_c[size].transpose((1, 0))

    pkl.dump((size2individual_runs_p, all_predicts), open(p_pkl_path, 'wb'))
    pkl.dump((size2individual_runs_c, all_predicts), open(c_pkl_path, 'wb'))
    pkl.dump((size2ensemble_runs_c, all_predicts), open(e_pkl_path, 'wb'))


# dumping path
data_name = 'sst-2'
p_pkl_path = os.path.join(p_dump_dir, data_name + '.pkl')
c_pkl_path = os.path.join(c_dump_dir, data_name + '.pkl')
e_pkl_path = os.path.join(e_dump_dir, data_name + '.pkl')

# loading template
f_path_template = 'model_data_for_release/sst-2/fold{fold}/results/s{s}p{p}f{f}epoch{i}over3.tsv'
data_path_template = 'model_data_for_release/sst-2/fold{fold}/data.json'

# dictionary that contains the results
size2individual_runs_c = {size: np.array([[[[] for ___ in range(4)] for __ in range(5)] for _ in range(10)]) for size
                        in model_sizes}
size2individual_runs_p = {size: np.array([[[[] for ___ in range(4)] for __ in range(5)] for _ in range(10)]) for size
                        in model_sizes}
size2ensemble_runs_c = {size: np.array([[] for _ in range(10)]) for size
                        in model_sizes}
all_predicts = []

for fold in range(5):
    data = json.load(open(data_path_template.format(fold=fold), 'r'))
    label_list = data['label_list']
    label_map = invert_labellist(label_list)
    predict_dicts = data['predict']
    all_predicts += predict_dicts

    for model_size in model_sizes:
        prob_4D, c_4D, ensemble_2D = [], [], []
        for pretrain_seed in range(1, 11):
            prob_3D, c_3D, probs = [], [], []
            for finetune_seed in range(1, 6):
                prob_2D, c_2D = [], []
                for i in range(9, 13):
                    f_path = f_path_template.format(p=pretrain_seed, f=finetune_seed, fold=fold, s=model_size, i=i)
                    print(f_path)
                    p = load_tsv(f_path)
                    if i == 12:
                        probs.append(p)
                    prob_2D.append(obtain_prob(p, predict_dicts, label_map).round(decimals=2))
                    c_2D.append(obtain_correctness(p, predict_dicts, label_map))
                print('acc', np.mean(prob_2D))
                prob_3D.append(prob_2D)
                c_3D.append(c_2D)

            probs_marginalized_f = np.mean(probs, axis=0)
            ensemble_2D.append(obtain_correctness(probs_marginalized_f, predict_dicts, label_map))
            prob_4D.append(prob_3D)
            c_4D.append(c_3D)

        size2individual_runs_p[model_size] = np.concatenate([size2individual_runs_p[model_size], prob_4D], axis=-1)
        size2individual_runs_c[model_size] = np.concatenate([size2individual_runs_c[model_size], c_4D], axis=-1)
        size2ensemble_runs_c[model_size] = np.concatenate([size2ensemble_runs_c[model_size], ensemble_2D], axis=-1)

for size in model_sizes:
    size2individual_runs_p[size] = size2individual_runs_p[size].transpose((3, 0, 1, 2))
    size2individual_runs_c[size] = size2individual_runs_c[size].transpose((3, 0, 1, 2))
    size2ensemble_runs_c[size] = size2ensemble_runs_c[size].transpose((1, 0))

pkl.dump((size2individual_runs_p, all_predicts), open(p_pkl_path, 'wb'))
pkl.dump((size2individual_runs_c, all_predicts), open(c_pkl_path, 'wb'))
pkl.dump((size2ensemble_runs_c, all_predicts), open(e_pkl_path, 'wb'))
