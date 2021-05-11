import numpy as np
import os
from collections import defaultdict
import json
from sklearn.gaussian_process import GaussianProcessRegressor
import random


lm_bucket = 'gs://bertbase/'
data_bucket = 'gs://data_resolution_logs/'
bsize = 10000


def uniform_density_sampling(X):
    bucket_id2sets = {}
    for idx, x in enumerate(X):
        key = int(x // 0.01)
        if key not in bucket_id2sets:
            bucket_id2sets[key] = set()
        bucket_id2sets[key].add(idx)
    result = []
    while len(bucket_id2sets) != 0 and len(result) < bsize:
        random_key = random.sample(bucket_id2sets.keys(), k=1)[0]
        random_idx = random.sample(bucket_id2sets[random_key], k=1)[0]
        bucket_id2sets[random_key].remove(random_idx)
        result.append(random_idx)
        if len(bucket_id2sets[random_key]) == 0:
            del bucket_id2sets[random_key]
    return result


def get_regressed_y(X_train, Y_train, X_pred):
    selected_idxes = uniform_density_sampling(X_train)
    gpr = GaussianProcessRegressor().fit(np.expand_dims(X_train[selected_idxes], axis=-1), Y_train[selected_idxes])
    num_pred = X_pred.shape[0]

    y_pred = np.array([])
    for i in range((num_pred - 1) // bsize + 1):
        x_in = X_pred[bsize * i:bsize * (i + 1)]
        y_pred = np.concatenate((y_pred, gpr.predict(x_in.reshape(-1, 1))))
    return y_pred

def get_binned_result(xs, ys):
    bins = [[] for _ in range(41)]
    for idx, x in enumerate(xs):
        bins[int(x / 0.025)].append(idx)
    result_xs = [np.mean(xs[b]) for b in bins]
    result_ys = [np.mean(ys[b]) for b in bins]
    return result_xs, result_ys


def cross_get_regressed_y(X, Y):
    orig_X = np.array(X)
    # shuffle and split
    num_all_data = X.shape[0]
    split_size1 = num_all_data // 2

    random_order = [_ for _ in range(num_all_data)]
    random.shuffle(random_order)
    reverse_order = np.argsort(random_order)
    X, Y = X[random_order], Y[random_order]
    X1, Y1, X2, Y2 = X[:split_size1], Y[:split_size1], X[split_size1:], Y[split_size1:]

    y1_pred, y2_pred = get_regressed_y(X2, Y2, X1), get_regressed_y(X1, Y1, X2)
    y_pred = np.concatenate((y1_pred, y2_pred))
    y_pred = y_pred[reverse_order]
    return y_pred


# Yes, the code for this part is ugly
# I don't want to fix it anyways :)
def calculate_diff_3D(c):
    c = c.astype(int)
    num_data, dim1, dim2 = c.shape

    result = []
    for d1 in range(dim1):
        for a in range(dim2):
            for b in range(a + 1, dim2):
                result.append(np.mean(np.abs(c[:, d1, a] - c[:, d1, b])))
    within_pretrain_diff = np.mean(result)

    result = []
    for a in range(dim1):
        for b in range(a + 1, dim1):
            for d in range(dim2):
                for e in range(dim2):
                    result.append(np.mean(np.abs(c[:, a, d] - c[:, b, e])))
    inter_pretrain_diff = np.mean(result)
    return within_pretrain_diff, inter_pretrain_diff




def invert_labellist(label_list):
    return {label: idx for idx, label in enumerate(label_list)}


def dict_arg_max(d):
    max_k = None
    for k in d:
        if max_k is None or d[k] > d[max_k]:
            max_k = k
    return max_k


def load_wrapper(f_path):
    import tensorflow as tf
    with tf.io.gfile.GFile(f_path, mode='r') as in_file:
        l = next(in_file)
        o = json.loads(l)
    return o


def write_wrapper(o, f_path):
    import tensorflow as tf
    with tf.io.gfile.GFile(f_path, mode='w') as out_file:
        s = json.dumps(o)
        out_file.write(s)


def load_tsv(f_path):
    ls = []
    with open(f_path, 'r') as in_file:
        for l in in_file:
            ls.append([float(f) for f in l.strip().split('\t')])
    return np.array(ls)


def load_all_results_from_dir(dir_name, reduction=None):
    steps2arr = {}
    for f_name in os.listdir(dir_name):
        if 'test_results' in f_name:
            step = f_name.replace('test_results', '').replace('.tsv', '')
            steps2arr[step] = load_tsv(os.path.join(dir_name, f_name))
    arrs = [steps2arr[step] for step in sorted(steps2arr.keys())]
    arrs = np.array(arrs)
    if reduction == 'mean':
        return np.mean(arrs, axis=0)
    if reduction == 'last':
        return arrs[-1]
    if type(reduction) == int:
        return np.mean(arrs[-reduction:], axis=0)
    return np.array(arrs)


def correctness(ground_truth_label, label_map, max_label_idx):
    if ground_truth_label != 'non-entailment':
        return label_map[ground_truth_label] == max_label_idx
    else:
        return max_label_idx != label_map['entailment']


def obtain_correctness(probabilities, input_dicts, label_map, return_type='list'):
    if return_type == 'dict':
        return {input_dicts[idx]['guid']: correctness(input_dicts[idx]['label'], label_map, np.argmax(probabilities[idx])) for idx in range(len(input_dicts))}
    if return_type == 'list':
        return np.array([correctness(input_dicts[idx]['label'], label_map, np.argmax(probabilities[idx])) for idx in range(len(input_dicts))])


def obtain_prediction(probabilities, labels):
    return [labels[np.argmax(p)] for p in probabilities]


def obtain_prob(probabilities, input_dicts, label_map):
    result = []
    for idx in range(len(input_dicts)):
        l = input_dicts[idx]['label']
        if l != 'non-entailment':
            result.append(probabilities[idx][label_map[l]])
        else:
            result.append(probabilities[idx][0] + probabilities[idx][2])
    return np.array(result)


def diff_fraction(correctness1, correctness2):
    return np.mean(np.array(correctness1) != np.array(correctness2))


def acc_by_group(correctness):
    groups = defaultdict(list)
    for datapoint_name, c in correctness.items():
        groups[datapoint_name.split('-')[0]].append(c)
    return {group_name: np.mean(groups[group_name]) for group_name in groups}


def pairwise_statistics_wrapper(f, arrs):
    num_arrs = len(arrs)
    agg = []
    for i in range(num_arrs):
        for j in range(i + 1, num_arrs):
            agg.append(f(arrs[i], arrs[j]))
    return np.mean(agg), np.std(agg)


def smooth(p):
    p = p + eps
    return p / np.sum(p)

eps = 1e-10
def KL(p1, p2):
    p1, p2 = smooth(p1), smooth(p2)
    distance = np.sum(p1 * np.log(p1 / p2))
    assert distance > 0
    return distance


def l2_distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


def get_cross_entropy_bias_variance_arrs(prob_arrs, labels):
    num_copies, num_data, num_labels = prob_arrs.shape
    pi_hat = np.exp(np.mean(np.log(prob_arrs), axis=0))
    pi_hat = pi_hat / np.expand_dims(np.sum(pi_hat, axis=-1), axis=1).repeat(3, 1)
    risk = np.mean([[-np.log(p[label]) for p, label in zip(prob_arrs[copy_idx], labels)] for copy_idx in range(num_copies)], axis=0)
    variance = np.array([np.mean([KL(pi_hat[idx], prob_arrs[copy_idx][idx]) for copy_idx in range(num_copies)]) for idx in range(num_data)])
    bias_squared = risk - variance
    return bias_squared, variance


def get_squared_bias_variance_arrs(prob_arrs, labels):
    num_copies, num_data, num_labels = prob_arrs.shape
    one_hot_labels = np.zeros((num_data, num_labels))
    one_hot_labels[np.arange(0, num_data, 1), labels] = 1
    pi_hat = np.mean(prob_arrs, axis=0)
    risk = np.mean([[l2_distance(p, label) for p, label in zip(prob_arrs[copy_idx], one_hot_labels)] for copy_idx in range(num_copies)], axis=0)
    variance = np.array([np.sum([l2_distance(prob_arrs[copy_idx][idx], pi_hat[idx]) for copy_idx in range(num_copies)]) / (num_copies - 1) for idx in range(prob_arrs.shape[1])])
    bias_squared = risk - variance
    return bias_squared, variance


def get_hard_squared_bias_variance_arrs(prob_arrs, labels):
    num_copies, num_data, num_labels = prob_arrs.shape
    hard_preds = np.zeros_like(prob_arrs)
    preds = np.argmax(prob_arrs, axis=-1)
    for copy_idx in range(num_copies):
        for idx in range(num_data):
            hard_preds[copy_idx][idx][preds[copy_idx][idx]] = 1
    return get_squared_bias_variance_arrs(hard_preds, labels)


pretrain_size2config = {
    'tiny': 'uncased_L-2_H-128_A-2',
    'mini': 'uncased_L-4_H-256_A-4',
    'small': 'uncased_L-4_H-512_A-8',
    'medium': 'uncased_L-8_H-512_A-8',
    'base': 'uncased_L-12_H-768_A-12',
    'large': 'wwm_uncased_L-24_H-1024_A-16'
}

def obtain_accuracy_from_dir(path):
    dataset = json.load(open(path.split('/result')[0] + '/data.json'))
    probs = load_all_results_from_dir(path, reduction='last')
    label_map = invert_labellist(dataset['label_list'])
    return np.mean(obtain_correctness(probs, dataset['predict'], label_map=label_map))


def split_into_groups(input_dicts):
    group_names = {d['guid'].split('-')[0] for d in input_dicts}
    group2idxes = {}
    for group in group_names:
        group2idxes[group] = [idx for idx in range(len(input_dicts)) if d['guid'].split('-')[0] == group_names]
    return group2idxes


def total_diff(a, b):
    return np.mean(np.abs(a - b))


def obsolete_boot_strap_4d(prob):
    assert len(prob.shape) == 4
    _, num_pretrain, num_finetune, _ = prob.shape
    pretraining_sampled = np.random.randint(0, num_pretrain, num_pretrain)
    finetuning_sampled = np.random.randint(0, num_finetune, size=(num_pretrain, num_finetune))
    pretrain_finetune_datapoint_iter = prob.transpose((1, 2, 0, 3))
    boot_strap = np.array(
        [pretrain_finetune_datapoint_iter[pretrain_seed][finetune_seeds].copy() for pretrain_seed, finetune_seeds in
         zip(pretraining_sampled, finetuning_sampled)])
    boot_strap = boot_strap.transpose((2, 0, 1, 3))
    return boot_strap


def boot_strap_4d(prob):
    assert len(prob.shape) == 4
    _, num_pretrain, num_finetune, num_iters = prob.shape
    pretraining_sampled = np.random.randint(0, num_pretrain, num_pretrain)
    finetuning_sampled = np.random.randint(0, num_finetune, size=(num_pretrain, num_finetune))
    iter_sampled = np.random.randint(0, num_iters, size=(num_pretrain, num_finetune, num_iters))
    pretrain_finetune_datapoint_iter = prob.transpose((1, 2, 3, 0))

    # boot_strap = np.array([[[[pretrain_finetune_datapoint_iter[pretraining_sampled][finetune_seed][iter_seed] for iter_seed in iter_sampled] for finetune_seed in finetuning_sampled] for pretrain_seed in pretraining_sampled]])
    boot_strap = np.array([[[pretrain_finetune_datapoint_iter[pretraining_sampled[pretrain_idx]][finetuning_sampled[pretrain_idx][finetune_idx]][iter_sampled[pretrain_idx][finetune_idx][iter_idx]]
                             for iter_idx in range(num_iters)] for finetune_idx in range(num_finetune)] for pretrain_idx in range(num_pretrain)])
    boot_strap = boot_strap.transpose((3, 0, 1, 2))
    return boot_strap

