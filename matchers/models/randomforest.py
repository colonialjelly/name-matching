import numpy as np
import Levenshtein  # NOTE: This library is GPL, but it's the only library I found to return the actual edit operations
from matchers import constant

# TODO add type hints


def get_editpairs(cand_name, name):
    """
    Compute the edit pairs: [(i,o)] where each (i,o) pair is the result of a levenshtein edit operation:
       replace (i, o),
       insert (i is empty), or
       delete (o is empty)
       i and o are replaced by their corresponding numeric indexes
    :param cand_name: source string
    :param name: target string
    :return: list of edit io pairs
    """
    editops = get_editops(cand_name, name)
    return list((char_to_idx_map[op[0]], char_to_idx_map[op[1]]) for op in editops)


def get_editops(src_name, tar_name):
    ops = Levenshtein.editops(src_name, tar_name)
    ops_pos, n1_pos, n2_pos = 0, 0, 0
    editops = []
    while True:
        if n1_pos == len(src_name) and n2_pos == len(tar_name):
            break
        c1 = src_name[n1_pos] if n1_pos < len(src_name) else ''
        c2 = tar_name[n2_pos] if n2_pos < len(tar_name) else ''
        op = ops[ops_pos] if ops_pos < len(ops) else None
        if op and op[1] == n1_pos and op[2] == n2_pos:
            if op[0] == 'replace':
                editops.append(c1+c2)
                n1_pos += 1
                n2_pos += 1
            elif op[0] == 'insert':
                editops.append('_'+c2)
                n2_pos += 1
            elif op[0] == 'delete':
                editops.append(c1+'_')
                n1_pos += 1
            else:
                raise Exception(f'Unexpected op {op}')
            ops_pos += 1
        else:
            editops.append(c1+c2)
            n1_pos += 1
            n2_pos += 1
    return editops


def get_Xy(editpairs_list, window_size):
    Xy = list((features, target) for editpairs in editpairs_list for features, target in
              get_pair_features_targets(editpairs, window_size))
    X, y = list(zip(*Xy))
    X = np.eye(vocab_size, dtype='u2')[np.array(X, dtype='u2')].reshape(len(X), -1)
    y = np.array(y, dtype='u2')
    return X, y


def get_pair_features_targets(editpairs, window_width=4):
    padding = [(0, 0)] * (window_width-1)
    features_targets = []
    editpairs = padding + editpairs
    for pos in range(window_width, len(editpairs)):
        features = list(char for pair in editpairs[pos-window_width:pos] for char in pair) + [editpairs[pos][0]]
        target = editpairs[pos][1]
        features_targets.append((features, target))
    return features_targets


# map edit pair characters to indexes and vice-versa
idx = range(2, constant.VOCAB_SIZE + 2)
char_to_idx_map = dict(zip(constant.ALPHABET, idx))
idx_to_char_map = dict(zip(idx, constant.ALPHABET))
# pad character
char_to_idx_map[' '] = 0
idx_to_char_map[0] = ' '
# insert/delete character
char_to_idx_map['_'] = 1
idx_to_char_map[1] = '_'
vocab_size = len(char_to_idx_map)


def chars_from_editpairs(editpairs):
    return list((idx_to_char_map[pair[0]], idx_to_char_map[pair[1]]) for pair in editpairs)


def get_features_targets(editops, window_width=4):
    padding = [0] * (window_width-1)
    features_targets = []
    editops = padding + editops + padding
    for pos in range(4, len(editops)-4):
        features = editops[pos-4:pos] + editops[pos+1:pos+5]
        target = editops[pos]
        features_targets.append((features, target))
    return features_targets
