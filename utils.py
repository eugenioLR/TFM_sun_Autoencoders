import numpy as np

def pearson(matrix, target):
    """
    Assume matrix is of size (samples, x, y)
    and target is of size (samples, 1)
    """
    
    avg_matrix = matrix.mean(axis=0)
    avg_target = target.mean(axis=0)
    
    avg_diff_mat = matrix - avg_matrix
    avg_diff_tar = target - avg_target
    
    term1 = np.sum(avg_diff_mat * avg_diff_tar, axis=0)
    term2 = np.sqrt(np.sum(avg_diff_mat**2, axis=0))
    term3 = np.sqrt(np.sum(avg_diff_tar**2, axis=0))
    
    term4 = np.maximum(term2*term3, 1e-4*np.ones(term2.shape))
    
    return term1/term4

def range_tuple(matrix):
    return matrix.min(), matrix.max()