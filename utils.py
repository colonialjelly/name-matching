import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def get_candidates_batch(X_input_names: np.ndarray,
                         X_source_names: np.ndarray,
                         names: np.ndarray,
                         num_candidates: int = 10,
                         metric='cosine',
                         normalized=False):
    """
    A function that computes scores between the input names and the source names using the given metric type.

    :param X_input_names: Vectorized input names of shape (m, k) where m is the number of input names and
                          k is the dimensionality of each vectorized name.
    :param X_source_names: Vectorized source names of shape (n, k)
    :param names: a nd.array that contains the actual string value of names
    :param num_candidates: Number of candidates to retrieve per name
    :param metric: Type of metric to use for fetching candidates
    :param normalized: Set it to true if X_input_names and X_source_names are L2 normalized
    :return: candidates: an nd.array of shape (m, num_candidates, 2)
    """
    sorted_scores_idx = None

    if metric == 'cosine':
        if normalized:  # If vectors are normalized dot product and cosine similarity are the same
            scores = safe_sparse_dot(X_input_names, X_source_names.T)
        else:
            scores = cosine_similarity(X_input_names, X_source_names)
    elif metric == 'euclidean':
        scores = euclidean_distances(X_input_names, X_source_names)
        sorted_scores_idx = np.argsort(scores, axis=1)[:, :num_candidates]
    else:
        raise ValueError("Unrecognized metric type. Valid options: 'dot_product', 'cosine', 'euclidean'")

    if sorted_scores_idx is None:
        sorted_scores_idx = np.flip(np.argsort(scores, axis=1), axis=1)[:, :num_candidates]

    sorted_scores = np.take_along_axis(scores, sorted_scores_idx, axis=1)
    ranked_candidates = names[sorted_scores_idx]
    candidates = np.dstack((ranked_candidates, sorted_scores))

    return candidates
