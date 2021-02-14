import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def get_candidates_batch(X_input_names: np.ndarray,
                         X_source_names: np.ndarray,
                         names: np.ndarray,
                         num_candidates: int = 10,
                         dist_type='dot_product'):
    """
    A function that computes scores according to a given distance/similarity function between the input batch of names
    and the all of the source names.

    :param X_input_names: Vectorized input names of shape (m, k) where m is the number of input names and
                          k is the dimensionality of each vectorized name.
    :param X_source_names: Vectorized source names of shape (n, k)
    :param names: a nd.array that contains the actual string value of names
    :param num_candidates: Number of candidates to retrieve per name
    :param dist_type: Type of distance measurement
    :return: candidates: an nd.array of shape (m, num_candidates, 2)
    """
    sorted_scores_idx = None

    if dist_type == 'dot_product':
        scores = safe_sparse_dot(X_input_names, X_source_names.T)
    elif dist_type == 'cosine':
        scores = cosine_similarity(X_input_names, X_source_names)
    elif dist_type == 'euclidean':
        scores = euclidean_distances(X_input_names, X_source_names)
        sorted_scores_idx = np.argsort(scores, axis=1)[:, 1:num_candidates+1]
    else:
        raise ValueError("Unrecognized similarity/distance type. Valid options: 'dot_product', 'cosine', 'euclidean'")

    if sorted_scores_idx is None:
        sorted_scores_idx = np.flip(np.argsort(scores, axis=1), axis=1)[:, 1:num_candidates+1]

    sorted_scores = np.take_along_axis(scores, sorted_scores_idx, axis=1)
    ranked_candidates = names[sorted_scores_idx]
    candidates = np.dstack((ranked_candidates, sorted_scores))

    return candidates

