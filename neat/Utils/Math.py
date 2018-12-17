import numpy as np
from scipy.stats import pearsonr

def find_non_orthogonal_columns(matrix):

    m_rank = np.linalg.matrix_rank(matrix)
    n_cols = matrix.shape[1]

    n_lin_dep = n_cols - m_rank

    cols_found = []

    for nc1 in range(n_cols):
        v1 = matrix[:,nc1]
        if np.sum(v1) == 0:
            cols_found.append(nc1)

            if len(cols_found) == n_lin_dep:
                return cols_found
        else:
            for nc2 in range(nc1+1,n_cols):
                v2 = matrix[:,nc2]
                if np.dot(v1.T,v2) != 0:
                    cols_found.append(nc1)

                    if len(cols_found) == n_lin_dep:
                        return cols_found
                    else:
                        break


    raise ValueError('Linearly independent columns have not been found')