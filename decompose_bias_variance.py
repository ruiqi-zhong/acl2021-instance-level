import numpy as np
import tqdm
import time


def sample_variance_1d(x):
    x = np.array(x)
    n_data = x.shape[0]
    mu = np.mean(x)
    result = np.sum((x - mu) ** 2) / (n_data - 1)
    return result


def batch_sample_variance_1d(x):
    x = np.array(x)
    n_data = x.shape[1]
    mu = np.mean(x, axis=-1)
    result = np.sum((x - np.expand_dims(mu, axis=-1).repeat(n_data, 1)) ** 2, axis=-1) / (n_data - 1)
    return result


def plugin_variance_1d(x):
    return np.std(x) ** 2


def variance_estimate_of_population_means(x):
    x = np.array(x)
    if len(x.shape) == 1:
        return sample_variance_1d(x)
    else:
        empirical_means = np.array([np.mean(s) for s in x])
        upward_biased_estimate = sample_variance_1d(empirical_means)
        mean_variance_estimate_of_mean_estimators = np.mean([variance_estimate_of_mean_estimator(s) for s in x])
        return upward_biased_estimate - mean_variance_estimate_of_mean_estimators


def batch_variance_estimate_of_population_means(x):
    x = np.array(x)
    if len(x.shape) == 2:
        return batch_sample_variance_1d(x)
    else:
        empirical_means = np.mean(x, axis=tuple(range(2, x.ndim)))
        upward_biased_estimate = batch_sample_variance_1d(empirical_means)
        mean_variance_estimate_of_mean_estimators = np.mean([batch_variance_estimate_of_mean_estimator(s) for s in np.swapaxes(x, 0, 1)], axis=0)
        return upward_biased_estimate - mean_variance_estimate_of_mean_estimators


def plugin_variance_estimate_of_population_means(x):
    x = np.array(x)
    empirical_means = [np.mean(s) for s in x]
    return plugin_variance_1d(empirical_means)


def variance_estimate_of_mean_estimator(x):
    x = np.array(x)
    if len(x.shape) == 1:
        n_data = x.shape[0]
        return sample_variance_1d(x) / n_data
    else:
        n_distributions = x.shape[0]
        part1 = np.mean([variance_estimate_of_mean_estimator(s) for s in x]) / n_distributions
        part2 = variance_estimate_of_population_means(x) / n_distributions
        return part1 + part2


def batch_variance_estimate_of_mean_estimator(x):
    x = np.array(x)
    if len(x.shape) == 2:
        n_data = x.shape[1]
        return batch_sample_variance_1d(x) / n_data
    else:
        n_distributions = x.shape[1]
        part1 = np.mean([batch_variance_estimate_of_mean_estimator(s) for s in np.swapaxes(x, 0, 1)], axis=0) / n_distributions
        part2 = batch_variance_estimate_of_population_means(x) / n_distributions
        return part1 + part2


def decompose_bias_variance(pred, y):
    decomposed_loss = []
    loss = np.mean((y - pred) ** 2)
    pred_shape = list(pred.shape)
    for var_dim in range(len(pred_shape)):
        reshaped_pred = np.reshape(pred, tuple([-1] + pred_shape[var_dim:]))
        decomposed_loss.append(np.mean([variance_estimate_of_population_means(s) for s in reshaped_pred]))
    return loss - np.sum(decomposed_loss), np.array(decomposed_loss)


def batch_decompose_bias_variance(pred, y):
    variances = []
    loss = np.mean((y - pred) ** 2, axis=tuple(range(1, pred.ndim)))
    pred_shape = list(pred.shape)
    for var_dim in range(len(pred_shape) - 1):
        reshaped_pred = np.reshape(pred, tuple([len(pred), -1] + pred_shape[var_dim + 1:]))
        vepms = np.array([batch_variance_estimate_of_population_means(s) for s in np.swapaxes(reshaped_pred, 0, 1)])
        variances.append(np.mean(vepms, axis=0))
    variances = np.array(variances).T
    return loss - np.sum(variances, axis=-1), variances



def plugin_decompose_bias_variance(pred, y):
    decomposed_loss = []
    loss = np.mean((y - pred) ** 2)
    pred_shape = list(pred.shape)
    for var_dim in range(len(pred_shape)):
        reshaped_pred = np.reshape(pred, tuple([-1] + pred_shape[var_dim:]))
        decomposed_loss.append(np.mean([plugin_variance_estimate_of_population_means(s) for s in reshaped_pred]))
    return loss - np.sum(decomposed_loss), np.array(decomposed_loss)






if __name__ == '__main__':
    """
    for size in np.arange(2, 15):
        print(size)
        mean_estimate = []
        xs = []
        for _ in range(50000):
            x = np.random.normal(0, 1, size=10)
            mean_estimate.append(np.mean(x))
            xs.append(variance_estimate_of_mean_estimator(x))
        mean_estimate = np.array(mean_estimate)
        xs = np.array(xs)
        print('=====')
        print(sample_variance_1d(mean_estimate))
        print(np.mean(xs))
        """
    """
    l = []
    mean_estimates = []
    var_of_mean_estimate = []
    for _ in range(50000):
        x = []
        for __ in range(3):
            m = np.random.normal()
            x.append(m + np.random.random(size=5))
        x = np.array(x)
        mean_estimates.append(np.mean(x))
        var_of_mean_estimate.append(variance_estimate_of_mean_estimator(x))
        v = variance_estimate_of_population_means(x)
        l.append(v)
    print(np.mean(l))
    print(sample_variance_1d(mean_estimates), np.mean(var_of_mean_estimate))
        """

    """
    from tqdm import trange
    biases_squared, variances = [], []
    for _ in trange(5000):
        x = []
        for __ in range(3):
            m = np.random.normal()
            x.append(m + np.random.random(size=5))
        x = np.array(x)
        y = 0.5
        bias_squared, variance = decompose_bias_variance(x, y)
        biases_squared.append(bias_squared)
        variances.append(variance)

    variances = np.array(variances)
    print(np.mean(biases_squared))
    print(np.mean(variances, axis=0))
        """
    from utils import boot_strap_4d, obsolete_boot_strap_4d
    test_number = 1

    # bootstrap and compare plugin estimator and unbiased estimator
    # average across estimator to test unbiasedness
    if test_number == 1:
        X = np.random.random(size=(1, 10, 5, 4))
        true_b, true_vs = plugin_decompose_bias_variance(X[0], 0)
        b_samples, vs_samples = [], []

        for _ in tqdm.trange(10000):
            # bs = obsolete_boot_strap_4d(X)
            bs = boot_strap_4d(X)
            b_sample, vs_sample = decompose_bias_variance(bs[0], 0)
            b_samples.append(b_sample)
            vs_samples.append(vs_sample)

        print(np.mean(b_samples), true_b)
        print(np.mean(vs_samples, axis=0), true_vs)

    # test whether the batch implementation is correct
    if test_number == 2:
        print('generating x')
        X = np.random.random(size=(200000, 10, 5, 4))
        print('x generated')

        s = time.time()
        batched_b, batched_v = batch_decompose_bias_variance(X, 0)
        w2 = time.time() - s
        print(w2)

        bs, vs = [], []
        s = time.time()
        for x in tqdm.tqdm(X):
            b, v = decompose_bias_variance(x, 0)
            bs.append(b)
            vs.append(v)
        w1 = time.time() - s
        bs, vs = np.array(bs), np.array(vs)

        print(bs - batched_b)
        print(vs - batched_v)
        print(w1, w2)

