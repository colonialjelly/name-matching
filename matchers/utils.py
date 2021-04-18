from typing import Union

import heapq
import jellyfish
import pandas as pd
import numpy as np
import torch
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from matchers import constant


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
        raise ValueError("Unrecognized metric type. Valid options: 'cosine', 'euclidean'")

    if sorted_scores_idx is None:
        sorted_scores_idx = np.flip(np.argsort(scores, axis=1), axis=1)[:, :num_candidates]

    sorted_scores = np.take_along_axis(scores, sorted_scores_idx, axis=1)
    ranked_candidates = names[sorted_scores_idx]
    candidates = np.dstack((ranked_candidates, sorted_scores))

    return candidates


def ndarray_to_exploded_df(candidates: np.ndarray, input_names: list, column_names: list):
    """
    Converts a 3d ndarray into an exploded pandas dataframe. Makes it easy to apply filters to the candidate set.
    :param candidates: Generated candidates for the given input names with shape (m, n, r)
    :param input_names: List of inputs names
    :param column_names: List of column names for the created dataframe
    :return: Pandas dataframe that has all the candidates in an exploded format
    """
    m, n, r = candidates.shape
    exploded_np = np.column_stack((np.repeat(input_names, n),
                                   candidates.reshape(m * n, -1)))
    exploded_df = pd.DataFrame(exploded_np, columns=column_names)
    return exploded_df


def convert_names_to_ids(names: np.ndarray, char_to_idx_map: dict, max_len: int):
    def convert_name(name):
        return [char_to_idx_map[c] for c in name]

    names_ids = list(map(convert_name, names))
    name_ids_chopped = [chop(name_id, max_len) for name_id in names_ids]
    name_ids_padded = [post_pad_to_length(name_id, max_len) for name_id in name_ids_chopped]
    return np.array(name_ids_padded)


def convert_ids_to_names(names_ids: np.ndarray, idx_to_char_map: dict):
    def convert_input_ids(input_ids):
        return ''.join([idx_to_char_map[input_id] for input_id in input_ids])

    names = list(map(convert_input_ids, names_ids))
    return np.array(names)


def post_pad_to_length(input_ids: Union[list, np.ndarray], length: int):
    num_tokens = len(input_ids)
    if num_tokens < length:
        pad_width = length - num_tokens
        return np.pad(input_ids, (0, pad_width), 'constant', constant_values=0)
    return np.array(input_ids)


def one_hot_encode(X: np.ndarray, vocab_length: int):
    return np.eye(vocab_length)[X]


def chop(tokens: Union[list, np.ndarray], max_length: int):
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens


def build_token_idx_maps():
    alphabet = list(constant.ALPHABET)
    idx = range(1, len(alphabet) + 1)
    char_to_idx_map = dict(zip(alphabet, idx))
    idx_to_char_map = dict(zip(idx, alphabet))

    char_to_idx_map[''] = 0
    idx_to_char_map[0] = ''

    return char_to_idx_map, idx_to_char_map


def remove_padding(name: str):
    return name[1:-1]


def add_padding(name: str):
    return constant.BEGIN_TOKEN + name + constant.END_TOKEN


def names_to_one_hot(names, char_to_idx_map, max_name_length):
    return check_convert_tensor(one_hot_encode(convert_names_to_ids(names, char_to_idx_map, max_name_length), constant.VOCAB_SIZE + 1))


def convert_names_model_inputs(names: Union[list, np.ndarray], char_to_idx_map: dict, max_name_length: int):
    X_targets = convert_names_to_ids(names, char_to_idx_map, max_name_length)
    X_one_hot = one_hot_encode(X_targets, constant.VOCAB_SIZE + 1)

    X_inputs = check_convert_tensor(X_one_hot)
    X_targets = check_convert_tensor(X_targets)

    return X_inputs, X_targets


def check_convert_tensor(X: Union[np.ndarray, torch.Tensor]):
    if not torch.is_tensor(X):
        return torch.from_numpy(X)
    return X


def get_k_near_negatives(name, positive_names, all_names, k):
    similarities = {}
    for cand_name in all_names:
        if cand_name != name and cand_name not in positive_names:
            dist = jellyfish.levenshtein_distance(name, cand_name)
            similarity = 1 - (dist / max(len(name), len(cand_name)))
            similarities[cand_name] = similarity
    return heapq.nlargest(k, similarities.keys(), lambda n: similarities[n])
