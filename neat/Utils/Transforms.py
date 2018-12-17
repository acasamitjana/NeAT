"""Module that defines useful functions to transform data and data-structures"""

import numpy as np

__threshold = (1e-14 ** 2)

def combinatorial(func, elements, k, start=0):
    """
    Generates func(x1, ...(func(x(k-2), func(x(k-1), xk))) for each possible
    combination of 'k' elements in 'elements' (repetitions are allowed).
    Example:
        func = lambda x, y: x + y # concatenate the strings
        elements = ['w', 'x', 'y', 'z']
        k = 3
        for c in combinatorial(func, elements, k):
            print c

        # The possible combinations of 3 elements taken from ['w', 'x', 'y', 'z']
        # (regardless of the order but allowing repetitions) will be printed, since
        # we just concatenated the elements selected by the function.
    """

    n = len(elements) - start
    if n > 0 and k > 0:
        for y in combinatorial(func, elements, k, start + 1):
            yield y
        x = elements[start]
        for d in range(1, k):
            for y in combinatorial(func, elements, k - d, start + 1):
                yield func(x, y)
            x = func(x, elements[start])
        yield x


def polynomial(degree, features, complete_polynomial=True, constant_term=False):
    if constant_term:
        assert len(features) > 0
        yield np.array([1] * len(features[0]))

    if not isinstance(features, np.ndarray):
        features = np.array(features)

    if complete_polynomial:
        init = 1
    else:
        init = degree

    for d in range(init, degree + 1):
        for term in combinatorial(lambda x, y: x * y, features, d):
            yield term


def copy_iterable(it):
    try:
        return (copy_iterable(x) for x in it)
    except TypeError:
        try:
            return it.copy()
        except AttributeError:
            return it


def tolist(it, first_call=True):
    try:
        return [tolist(x, False) for x in it]
    except TypeError:
        if first_call:
            return [it]
        else:
            return it


def orthogonalize_all(features):
    '''Orthogonalizes each feature w.r.t the others

        Modifies:

            - Features: each column has been orthogonalized with respect to the previous ones.

        Returns:

            - Deorthogonalization matrix: A (C+R)x(C+R) (2-dimensional) upper triangular matrix that yields the
                original 'feature' matrix when right-multiplied with the new 'feature'
                matrix. More specifically, given the original 'feature' matrix, OF, and the new, orthogonalized
                'correctors' matrix, NF,  and the return value is a matrix, D,
                such that OF = NF x D (matrix multiplication).
    '''

    # Original 'features' matrix:
    #     V = (C | R) = ( v_1 | v_2 | ... | v_(C+R) )

    # Gram-Schmidt:
    #    u_j = v_j - sum_{i < j} ( ( < u_i, v_j > / < u_i, u_i > ) * u_i ) # orthogonalize v_j with respect to every u_i, or equivalently, v_i, with i < j

    # New 'features' matrix (orthonormalized):
    #    U = ( u_1 | u_2 | ... | u_(C+R) )

    # Deorthogonalization matrix (upper triangular):
    #    D[i, j] =
    #            < u_i, v_j > / < u_i, u_i >,    if i < j
    #             1,                                if i = j
    #             0,                                if i > j
    new_features = np.zeros_like(features)

    C = features.shape[1]
    D = np.zeros((C, C))  # D[i, j] = 0, if i > j
    if (C == 0):
        return D

    threshold = features.shape[0] * __threshold

    for i in range(C - 1):
        D[i, i] = 1.0  # D[i, j] = 1, if i = j

        u_i = features[:, i]
        norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

        if norm_sq < threshold:
            u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
            # Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
            # work, hopefully with a small enough precision error)
            continue

        for j in range(i + 1, C):  # for j > i
            v_j = features[:, j]

            D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
            v_j -= D[i, j] * u_i  # Orthogonalize v_j with respect to u_i (Gram-Schmidt, iterating over j instead of i)
            features[:,j] = v_j

        new_features[:,i] = u_i

    D[-1, -1] = 1.0  # D[i, j] = 1, if i = j

    return new_features, D

def normalize_all(features):
    '''Normalizes the energy of each corrector (the magnitude of each feature interpreted as a vector,
            that is, the magnitude of each column of the internal correctors matrix).

            Modifies:

                - Correctors: each column has been normalized to have unit magnitude.

            Returns:

                - Denormalization matrix: A CxC (2-dimensional) diagonal matrix that yields the original
                    'correctors' matrix when right-multiplied with the new 'correctors' matrix. That is,
                    given the original 'correctors' matrix, OC, and the new, normalized 'correctors' matrix,
                    NC, the return value is a diagonal matrix D such that OC = NC x D (matrix multiplication).
    '''

    # Original 'correctors' matrix:
    #    V = ( v_1 | v_2 | ... | v_C )

    # Normalization:
    #    u_j = v_j / ||v_j||

    # New 'correctors' matrix (normalized):
    #    U = ( u_1 | u_2 | ... | u_C )

    # Deorthogonalization matrix (diagonal):
    #    D[i, j] =
    #             ||u_i||,    if i = j
    #             0,            if i != j

    new_features = np.zeros_like(features)

    C = features.shape[1]
    D = np.zeros((C, C))  # D[i, j] = 0, if i != j

    threshold = features.shape[0] * __threshold

    for i in range(C):
        u_i = features[:, i]
        norm_sq = u_i.dot(u_i)
        if norm_sq >= threshold:
            D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
            u_i /= D[i, i]  # Normalization
        elif norm_sq != 0.0:
            u_i[:] = 0.0

        new_features[:, i] = u_i

    return new_features, D

def orthonormalize_all(features):
    '''Orthogonalizes each predictor w.r.t the others, all correctors w.r.t. the others, and all the
        predictors w.r.t. all the correctors, and normalizes the results. This is equivalent to applying
        orthogonalize_all and normalize_all consecutively (in that same order), but slightly faster.

        Modifies:

            - Correctors: each column has been orthogonalized with respect to the previous np.ones and nor-
                malized afterwards.
            - Regressors: each column has been orthogonalized with respect to all the columns in the
                correctors matrix and all the previous columns in the predictors matrix, and normalized
                afterwards.

        Returns:

            - Deorthonormalization matrix: A (C+R)x(C+R) (2-dimensional) upper triangular matrix that yields
                the original 'correctors' and 'predictors' matrices when right-multiplied with the new
                'correctors and 'predictors' matrices. More specifically, given the original 'correctors'
                matrix, namely OC, the original 'predictors' matrix, OR, and the new, orthonormalized
                'correctors' and 'predictors' matrices, NC and NR respectively, the return value is a matrix,
                D, such that (OC | OR) = (NC | NR) x D (matrix multiplication).
    '''

    # Original 'features' matrix:
    #     V = (C | R) = ( v_1 | v_2 | ... | v_(C+R) )

    # Gram-Schmidt:
    #    u_j = v_j - sum_{i < j} ( < w_i, v_j > * w_i ) # orthogonalize v_j with respect to w_i, or equivalently, u_i or v_i with i < j
    #    w_j = u_j / (||u_j||) = u_j / sqrt(< u_j, u_j >) # normalize u_j

    # New 'features' matrix (orthonormalized):
    #    W = ( w_1 | w_2 | ... | w_(C+R) )

    # Deorthonormalization matrix (upper triangular):
    #    D[i, j] =
    #            < w_i, v_j >,        if i < j
    #             ||u_i||,            if i = j
    #             0,                    if i > j
    new_features = np.zeros_like(features)

    C = features.shape[1]
    D = np.zeros((C, C))  # D[i, j] = 0, if i > j
    if (C == 0):
        return D

    threshold = features.shape[0] * __threshold

    for i in range(C - 1):
        u_i = features[:, i]
        norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

        if norm_sq < threshold:
            u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
            # Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
            # work, hopefully with a small enough precision error)
            continue

        D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
        u_i /= D[i, i]  # Normalize u_i, now u_i denotes w_i (step 2 of Gram-Schmidt)
        for j in range(i + 1, C):  # for j > i
            v_j = features[:, j]

            D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
            v_j -= D[i, j] * u_i  # Orthogonalize v_j with respect to u_i (Gram-Schmidt, iterating over j instead of i)
            features[:, j] = v_j

        new_features[:, i] = u_i

    D[-1, -1] = 1.0  # D[i, j] = 1, if i = j

    return new_features, D
