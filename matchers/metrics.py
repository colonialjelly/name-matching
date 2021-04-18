import numpy as np
import math
import matplotlib.pyplot as plt
import scipy


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


def precision_recall(relevants, candidates, N):
    precisions = []
    recalls = []
    for i in range(N):
        precisions.append(np.mean([precision_k(a, c, i + 1) for a, c in zip(relevants, candidates)]))
        recalls.append(np.mean([recall_k(a, c, i + 1) for a, c in zip(relevants, candidates)]))
    return precisions, recalls


def precision_recall_curve(relevants, candidates, N):
    show_precision_recall_curve(*precision_recall(relevants, candidates, N))


def show_precision_recall_curve(precisions, recalls):
    plt.plot(recalls, precisions, 'ko--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def precision_at_threshold(weighted_actual, candidate, threshold):
    matches = candidate[candidate[:,1] >= threshold][:,0]
    num_matches = len(matches)
    if num_matches == 0:
        return 1.0
    return len(set(name for name, weight, _ in weighted_actual).intersection(matches)) / num_matches


def recall_at_threshold(weighted_actual, candidate, threshold):
    matches = candidate[candidate[:,1] >= threshold][:,0]
    return sum(weight for name, weight, _ in weighted_actual if name in matches)


def avg_precision_at_threshold(weighted_actuals, candidates, threshold):
    avg_precisions = []
    for a, c in zip(weighted_actuals, candidates):
        avg_precisions.append(precision_at_threshold(a, c, threshold))
    return np.mean(avg_precisions)


def avg_recall_at_threshold(weighted_actuals, candidates, threshold):
    avg_recalls = []
    for a, c in zip(weighted_actuals, candidates):
        avg_recalls.append(recall_at_threshold(a, c, threshold))
    return np.mean(avg_recalls)


def precision_recall_curve_at_threshold(weighted_actuals, candidates, min_threshold=0.5, max_threshold=1.0, step=0.01):
    show_precision_recall_curve(*precision_recall_at_threshold(weighted_actuals, candidates, min_threshold, max_threshold, step))


def precision_recall_at_threshold(weighted_actuals, candidates, min_threshold=0.5, max_threshold=1.0, step=0.01):
    precisions = []
    recalls = []
    for i in np.arange(min_threshold, max_threshold, step):
        precisions.append(np.mean([precision_at_threshold(a, c, i) for a, c in zip(weighted_actuals, candidates)]))
        recalls.append(np.mean([recall_at_threshold(a, c, i) for a, c in zip(weighted_actuals, candidates)]))
    return precisions, recalls


def get_auc(weighted_relevant_names_test, candidates, min_threshold=0.5, max_threshold=1.0, step=0.01):
    precisions, recalls = precision_recall_at_threshold(weighted_relevant_names_test, candidates, min_threshold, max_threshold, step)
    # take recall all the way to 0 with last precision so the curve starts at 0
    precisions.append(precisions[-1])
    recalls.append(0.0)
    precs = []
    recs = []
    prev_r = 0
    for p, r in zip(precisions, recalls):
        if math.isclose(r, prev_r):
            continue
        precs.append(p)
        recs.append(r)
        prev_r = r
    precs.reverse()
    recs.reverse()
    return scipy.integrate.simpson(precs, recs)