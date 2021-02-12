import numpy as np


def precision_k(actual, candidates, k):
    if len(candidates) > k:
        candidates = candidates[:k]
    return len(set(actual).intersection(candidates)) / min(k, len(candidates))


def avg_precision_k(actual, candidates, max_k):
    precisions = []
    for i in range(max_k):
        precisions.append(precision_k(actual, candidates, i + 1))

    return np.mean(precisions), precisions


def mean_avg_precision_k(actuals, candidates, max_k):
    avg_precisions = []
    for a, c in zip(actuals, candidates):
        avg_precisions.append(avg_precision_k(a, c, max_k)[0])

    return np.mean(avg_precisions)


def recall_k(actual, candidates, k):
    if len(candidates) > k:
        candidates = candidates[:k]
    return len(set(actual).intersection(candidates)) / len(actual)