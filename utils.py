import numpy as np

def pearson_mat(matrix, target):
    """
    Assume matrix is of size (samples, x, y)
    and target is of size (samples, 1)
    """
    if len(target.shape) < 3:
        target.shape += (1,) * (3-len(target.shape))
    avg_matrix = matrix.mean(axis=0)
    avg_target = target.mean(axis=0)
    avg_diff_mat = matrix - avg_matrix
    avg_diff_tar = target - avg_target
    term1 = np.sum(avg_diff_mat * avg_diff_tar, axis=0)
    term2 = np.sqrt(np.sum(avg_diff_mat**2, axis=0))
    term3 = np.sqrt(np.sum(avg_diff_tar**2, axis=0))
    term4 = np.maximum(term2*term3, 1e-4*np.ones(term2.shape))
    return term1/term4


def spearman_mat(matrix, target):
    rank_x = np.argsort(matrix, axis=0)
    rank_y = np.argsort(target, axis=0)
    return pearson_mat(rank_x, rank_y)


def range_tuple(matrix):
    return matrix.min(), matrix.max()