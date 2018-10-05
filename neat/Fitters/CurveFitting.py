from abc import ABCMeta

import numpy as np
import os

from neat.FitScores.FitEvaluation import evaluation_function as eval_func
from neat.Utils.Documentation import docstring_inheritor, Debugger



class abstractstatic(staticmethod):
    __slots__ = ()
    __isabstractmethod__ = True

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True


class CurveFitter(object):
    '''Abstract class that implements the framework to develop curve fitting algorithms.
    '''
    __metaclass__ = docstring_inheritor(ABCMeta)
    __threshold = (1e-14 ** 2)

    IncludeIntercept = False


    def __init__(self, covariates=None, intercept=IncludeIntercept):
        '''[Constructor]

            Parameters:

                - covariates: NxC (2-dimensional) matrix (default None), representing the covariates,
                    i.e., features that (may) explain a part of the observational data in which we are
                    not interested, where C is the number of covariates and N the number of elements
                    for each covariate.

                - intercept: one of CurveFitter.NoIntercept, CurveFitter.PredictionIntercept or
                    CurveFitter.CorrectionIntercept (default CurveFitter.NoIntercept), indicating whether
                    the intercept (a.k.a. homogeneous term) must be incorporated to the model or not, and
                    if so, wheter it must be as a predictor or a corrector. In the last two cases, a column
                    of ones will be added as the first column (feature) of the internal predictors/correctors
                    matrix. Please notice that, if the matrix to which the columns of ones must precede
                    does not have any elements, then this parameter will have no effect.
        '''


        if not covariates is None:
            correctors = np.array(covariates, dtype=np.float64)

            if len(correctors.shape) != 2:
                raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix (or None)')

        if intercept:
            self._crvfitter_covariates = np.concatenate((np.ones((covariates.shape[0], 1)), covariates),axis=1)
        else:
            self._crvfitter_covariates = covariates

        C = self._crvfitter_covariates.shape[1]
        self._crvfitter_covariates_parameters = np.zeros((C, 0))

    @property
    def covariates(self):
        '''Matrix of shape (N, C), representing the correctors of the model.
        '''
        return self._crvfitter_covariates.copy()

    @property
    def covariates_parameters(self):
        '''Array-like structure of shape (Kc, X1, ..., Xn), representing the correction parameters for which
            the correctors best explain the observational data passed as argument in the last call to 'fit',
            where Kc is the number of parameters for each variable in such observations, and X1, ..., Xn are
            the dimensions of the 'observations' argument in the last call to 'fit' (there are X1*...*Xn
            variables).
        '''
        return self._crvfitter_covariates_parameters.copy().reshape(-1, *self._crvfitter_dims)


    def orthogonalize_covariates(self):
        '''Orthogonalizes each corrector in the structure w.r.t. all the previous ones. That is, for each
            column in the correctors matrix, its projection over the previous columns is computed and sub-
            tracted from it.

            Modifies:

                - Correctors: each column has been orthogonalized with respect to the previous np.ones.

            Returns:

                - Deorthogonalization matrix: A CxC (2-dimensional) upper triangular matrix that yields the
                    original 'correctors' matrix when right-multiplied with the new 'correctors' matrix. That
                    is, given the original 'correctors' matrix, OC, and the new, orthogonalized 'correctors'
                    matrix, NC, the return value is a matrix, D, such that OC = NC x D (matrix multiplication).
        '''

        # Original 'correctors' matrix:
        #     V = ( v_1 | v_2 | ... | v_C )

        # Gram-Schmidt:
        #    u_j = v_j - sum_{i < j} ( ( < u_i, v_j > / < u_i, u_i > ) * u_i ) # orthogonalize v_j with respect to every u_i, or equivalently, v_i, with i < j

        # New 'correctors' matrix (orthonormalized):
        #    U = ( u_1 | u_2 | ... | u_C )

        # Deorthogonalization matrix (upper triangular):
        #    D[i, j] =
        #            < u_i, v_j > / < u_i, u_i >,    if i < j
        #             1,                                if i = j
        #             0,                                if i > j

        C = self._crvfitter_covariates.shape[1]
        D = np.zeros((C, C))  # D[i, j] = 0, if i > j
        if (C == 0):
            return D

        threshold = self._crvfitter_covariates.shape[0] * CurveFitter.__threshold

        for i in range(C - 1):
            D[i, i] = 1.0  # D[i, j] = 1, if i = j

            u_i = self._crvfitter_covariates[:, i]
            norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

            if norm_sq < threshold:
                u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
                # Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
                # work, hopefully with a small enough precision error)
                continue

            for j in range(i + 1, C):  # for j > i
                v_j = self._crvfitter_covariates[:, j]

                D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
                v_j -= D[
                           i, j] * u_i  # Orthogonalize v_j with respect to u_i (Gram-Schmidt, iterating over j instead of i)

        D[-1, -1] = 1.0  # D[i, j] = 1, if i = j

        return D

    def normalize_covariates(self):
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

        C = self._crvfitter_covariates.shape[1]
        D = np.zeros((C, C))  # D[i, j] = 0, if i != j

        threshold = self._crvfitter_covariates.shape[0] * CurveFitter.__threshold

        for i in range(C):
            u_i = self._crvfitter_covariates[:, i]
            norm_sq = u_i.dot(u_i)
            if norm_sq >= threshold:
                D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
                u_i /= D[i, i]  # Normalization
            elif norm_sq != 0.0:
                u_i[:] = 0.0

        return D

    def orthonormalize_covariates(self):
        '''Orthogonalizes each corrector with respect to all the previous ones, and normalizes the results.
            This is equivalent to applying orthogonalize_correctors and normalize_correctors consecutively
            (in that same order), but slightly faster.

            Modifies:

                - Correctors: each column has been orthogonalized w.r.t. the previous np.ones, and normalized
                    afterwards.

            Returns:

                - Deorthonormalization matrix: A CxC (2-dimensional) upper triangular matrix that yields the
                    original 'correctors' matrix when right-multiplied with the new 'correctors' matrix. That
                    is, given the original 'correctors' matrix, OC, and the new, orthonormalized 'correctors'
                    matrix, NC, the return value is a matrix, D, such that OC = NC x D (matrix multiplication).
        '''

        # Original 'correctors' matrix:
        #     V = ( v_1 | v_2 | ... | v_C )

        # Gram-Schmidt:
        #    u_j = v_j - sum_{i < j} ( < w_i, v_j > * w_i ) # orthogonalize v_j with respect to w_i, or equivalently, u_i or v_i with i < j
        #    w_j = u_j / (||u_j||) = u_j / sqrt(< u_j, u_j >) # normalize u_j

        # New 'correctors' matrix (orthonormalized):
        #    W = ( w_1 | w_2 | ... | w_C )

        # Deorthonormalization matrix (upper triangular):
        #    D[i, j] =
        #            < w_i, v_j >,        if i < j
        #             ||u_i||,            if i = j
        #             0,                    if i > j

        C = self._crvfitter_covariates.shape[1]
        D = np.zeros((C, C))  # D[i, j] = 0, if i > j

        threshold = self._crvfitter_covariates.shape[0] * CurveFitter.__threshold

        for i in range(C):
            u_i = self._crvfitter_covariates[:, i]

            norm_sq = u_i.dot(u_i)  # < u_i, u_i > = ||u_i||**2
            if norm_sq < threshold:
                u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
                # Notice that D[i, i] is set to 0, which is exactly the same as ||u_i||, as requested (this means that
                # the deorthonormalization will still work, hopefully with a small enough precision error)
                continue

            D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
            u_i /= D[i, i]  # Normalize u_i, now u_i denotes w_i (step 2 of Gram-Schmidt)

            for j in range(i + 1, C):  # for j > i
                v_j = self._crvfitter_covariates[:, j]

                D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
                v_j -= D[
                           i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

        return D


    @abstractstatic
    def __fit__(covariates, observations, *args, **kwargs):
        '''[Abstract method] Computes the correction and prediction parameters that best fit the observations.
            This method is not intended to be called outside the CurveFitter class.

            Parameters:

                - correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that
                    (may) explain a part of the observational data in which we are not interested, where C
                    is the number of correctors and N the number of elements for each corrector.

                - predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be
                    used to try to explain/predict the observations (experimental data), where R is the number
                    of predictors and N the number of elements for each predictor (the latter is ensured to be
                    the same as that in the 'correctors' argument).

                - observations: NxM (2-dimensional) matrix, representing the observational data, i.e., values
                    obtained by measuring the variables of interest, whose behaviour is wanted to be explained
                    by the correctors and predictors, where M is the number of variables and N the number of
                    observations for each variable (the latter is ensured to be the same as those in the
                    'correctors' and the 'predictors' arguments).

                - any other arguments will also be passed to the method in the subclass.

            Returns:

                - Correction parameters: (Kc)xM (2-dimensional) matrix, representing the parameters that best
                    fit the correctors to the observations for each variable, where M is the number of variables
                    (same as that in the 'observations' argument) and Kc is the number of correction parameters
                    for each variable.

                - Regression parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best
                    fit the predictors to the corrected observations for each variable, where M is the number
                    of variables (same as that in the 'observations' argument) and Kr is the number of
                    prediction parameters for each variable.


            [Developer notes]
                - Assertions regarding the size and type of the arguments have already been performed before
                    the call to this method to ensure that the sizes of the arguments are coherent and the
                    observations matrix has at least one element.

                - The 'correctors' and 'predictors' matrices may have zero elements, in which case the behaviour
                    of the method is left to be decided by the subclass.

                - You may modify the 'observations' matrix if needed, but both the 'correctors' and the
                    'predictors' arguments should be left unchanged.

                - The result should be returned as a tuple of 2 elements, containing the correction parameters
                    in the first position and the prediction parameters in the second position.

                - Although it is defined as a static method here, this method supports a non-static implementation.
        '''
        raise NotImplementedError

    def fit(self, observations, *args, **kwargs):
        '''Computes the correction and prediction parameters that best fit the observations.

            Parameters:

                - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational
                    data, i.e., values obtained by measuring the variables of interest, whose behaviour is wanted
                    to be explained by the correctors and predictors in the system, where M = X1*...*Xn is the
                    number of variables and N the number of observations for each variable.

                - any other arguments will be passed to the __fit__ method.

            Modifies:

                - [created] Correction parameters: array-like structure of shape (Kc, X1, ..., Xn), representing
                    the parameters that best fit the correctors to the observations, where X1, ..., Xn are the
                    original dimensions of the 'observations' argument and Kc is the number of correction parameters
                    for each variable.

                - [created] Regression parameters: array-like structure of shape (Kr, X1, ..., Xn), representing
                    the parameters that best fit the predictors to the observations, where X1, ..., Xn are the
                    original dimensions of the 'observations' argument and Kr is the number of prediction parameters
                    for each variable.
        '''
        obs = np.array(observations, dtype=np.float64)
        dims = obs.shape
        print(dims)
        print(self._crvfitter_covariates.shape)
        self._crvfitter_dims = dims[1:]
        if dims[0] != self._crvfitter_covariates.shape[0]:
            raise ValueError('Observations and features (correctors and/or predictors) have incompatible sizes')

        if 0 in dims:
            raise ValueError('There are no elements in argument \'observations\'')

        obs = obs.reshape(dims[0], -1)
        self._crvfitter_correction_parameters, self._crvfitter_prediction_parameters = self.__fit__(
            self._crvfitter_covariates, obs, *args, **kwargs)
        return self

    @abstractstatic
    def __predict__(covariates, covariates_parameters, *args, **kwargs):
        '''[Abstract method] Computes a prediction using the predictors together with the prediction parameters.
            This method is not intended to be called outside the CurveFitter class.

            Parameters:

                - predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
                    to try to explain/predict the observations (experimental data), where R is the number of
                    predictors and N the number of elements for each predictor.

                - prediction_parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best fit
                    the predictors to the corrected observations for each variable, where M is the number of
                    variables and Kr is the number of prediction parameters for each variable.

                - any other arguments will also be passed to the method in the subclass.

            Returns:

                - Prediction: NxM (2-dimensional) matrix, containing N predicted values for each of the M variables.

            [Developer notes]
                - Assertions regarding the size and type of the arguments have already been performed before the
                    call to this method to ensure that the sizes of the arguments are coherent and both, the
                    'predictors' and the 'prediction_parameters' matrices have at least one element each.

                - Both the 'predictors' and the 'prediction_parameters' arguments should be left unchanged.

                - Although it is defined as a static method here, this method supports a non-static implementation.
        '''
        raise NotImplementedError

    def predict(self, covariates=None, covariates_parameters=None, *args, **kwargs):
        '''Computes a prediction using the predictors together with the prediction parameters.

            Parameters:

                - predictors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
                    to be used to try to explain/predict the observations (experimental data), where R is the number
                    of predictors and N the number of elements for each predictor. If set to None, the predictors of
                    the instance will be used.

                - prediction_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
                    the parameters to fit the predictors to the corrected observations for each variable, where M =
                    X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
                    variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
                    used.

                - any other arguments will be passed to the __predict__ method.

            Returns:

                - Prediction: array-like structure of shape (N, X1, ..., Xn), containing N predicted values for each of
                    the M = X1*...*Xn variables.
        '''
        if covariates is None:
            preds = self._crvfitter_covariates
            if 0 in preds.shape:
                raise AttributeError('There are no predictors in this instance')
        else:
            preds = np.array(covariates, dtype=np.float64)
            if len(preds.shape) != 2:
                raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')
            if 0 in preds.shape:
                raise ValueError('There are no elements in argument \'predictors\'')

        if covariates_parameters is None:
            params = self._crvfitter_covariates_parameters
            dims = (1,) + self._crvfitter_dims
        else:
            params = np.array(covariates_parameters, dtype=np.float64)
            # Keep original dimensions (to reset dimensions of prediction)
            dims = params.shape
            # Make matrix 2-dimensional
            params = params.reshape(dims[0], -1)

        if 0 in dims:
            raise ValueError('There are no elements in argument \'prediction_parameters\'')

        prediction = self.__predict__(preds, params, *args, **kwargs)

        # Restore original dimensions (except for the first axis)
        return prediction.reshape(-1, *dims[1:])

    def evaluate_fit(self, observations, evaluation_function, covariates=None, covariates_parameters=None,
                     *args, **kwargs):
        """Evaluates the degree to which the correctors and predictors get to explain the observational
            data passed in the 'observations' argument by using the evaluator at 'evaluation_function'.
    
            Parameters:
    
                - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational data,
                    i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
                    explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
                    variables and N the number of observations for each variable.
    
                - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
                    that (may) explain a part of the observational data in which we are not interested, where C is
                    the number of correctors and N the number of elements for each corrector. If set to None, the
                    internal correctors will be used.
    
                - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
                    the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
                    is the number of variables and Kc the number of correction parameters for each variable. If set
                    to None, the correction parameters obtained in the last call to 'fit' will be used.
    
                - predictors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
                    to be used to try to explain/predict the observations (experimental data), where R is the number
                    of predictors and N the number of elements for each predictor. If set to None, the internal re-
                    gressors will be used.
    
                - prediction_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
                    the parameters to fit the predictors to the corrected observations for each variable, where M =
                    X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
                    variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
                    used.
    
                - any other arguments will be passed to the 'evaluation_function.evaluate' method.
    
            Returns:
    
                - Fitting scores: array-like structure of shape (X1, ..., Xn), containing floats that indicate the
                    goodness of the fit, that is, how well the predicted curves represent the corrected observational
                    data, or equivalently, how well the model applied to the predictors explains the observed data.
        """

        obs = np.array(observations, dtype=np.float64)
        dims = obs.shape
        obs = obs.reshape(dims[0], -1)

        if 0 in dims:
            raise ValueError('There are no elements in argument \'observations\'')

        if covariates is None:
            cors = self._crvfitter_covariates
            if 0 in cors.shape:
                correctors_present = False
            else:
                correctors_present = True
        else:
            cors = np.array(covariates, dtype=np.float64)
            if len(cors.shape) != 2:
                raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')

            if 0 in cors.shape:
                raise ValueError('There are no elements in argument \'correctors\'')

            correctors_present = True

        if correctors_present:
            if obs.shape[0] != cors.shape[0]:
                raise ValueError('The dimensions of the observations and the correctors are incompatible')

            if covariates_parameters is None:
                cparams = self._crvfitter_correction_parameters
                if 0 in cparams.shape:
                    raise AttributeError('There are no correction parameters in this instance')
            else:
                cparams = np.array(covariates_parameters, dtype=np.float64)
                cparams = cparams.reshape(cparams.shape[0], -1)

                if 0 in cparams.shape:
                    raise ValueError('There are no elements in argument \'correction_parameters\'')

            if obs.shape[1] != cparams.shape[1]:
                raise ValueError('The dimensions of the observations and the correction parameters are incompatible')

        else:
            cparams = np.zeros((0, 0))


        if obs.shape[0] != covariates.shape[0]:
            raise ValueError('The dimensions of the observations and the covariates are incompatible')

        if covariates_parameters is None:
            pparams = self._crvfitter_covariates_parameters
            if 0 in pparams.shape:
                raise AttributeError('There are no covariates parameters in this instance')
        else:
            pparams = np.array(covariates_parameters, dtype=np.float64)
            # Make matrix 2-dimensional
            pparams = pparams.reshape(pparams.shape[0], -1)

            if 0 in pparams.shape:
                raise ValueError('There are no elements in argument \'prediction_parameters\'')

        if obs.shape[1] != pparams.shape[1]:
            raise ValueError('The dimensions of the observations and the prediction parameters are incompatible')

        class FittingResults(object):
            pass

        fitres = FittingResults()

        fitres.observations = obs
        fitres.covariates = covariates
        fitres.covariates_parameters = cparams

        fitting_scores = evaluation_function[self].evaluate(fitres, *args, **kwargs)

        return fitting_scores.reshape(dims[1:])

    def __df_fitting__(self, observations, covariates, covariates_parameters):
        '''
        Computes the (effective) degrees of freedom for the fitted predictors.
        This method is not intended to be called outside the CurveFitter class.

        Parameters
        ----------

        observations : numpy.array
            array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
            i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
            explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
            variables and N the number of observations for each variable.

        predictors : numpy.array
            NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
            to try to explain/predict the observations (experimental data), where R is the number of
            predictors and N the number of elements for each predictor.

        prediction_parameters : numpy.array
            (Kr)xM (2-dimensional) matrix, representing the parameters that best fit
            the predictors to the corrected observations for each variable, where M is the number of
            variables and Kr is the number of prediction parameters for each variable.

        Returns
        -------

        numpy.array
            The (effective) degrees of freedom for the fitted predictors for each variable
            X1, X2, ..., Xn
        '''
        raise NotImplementedError

    def df_fitting(self, observations, covariates=None, covariates_parameters=None):
        '''
        Computes the (effective) degrees of freedom for the fitted predictors

        Parameters
        ----------

        observations : numpy.array
            array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
            i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
            explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
            variables and N the number of observations for each variable. Be aware that the observations
            must be the same as the ones used in prediction, that is, if the prediction was performed over
            corrected observations, these corrected observations are the ones that must be used here.

        predictors : numpy.array
            NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
            to be used to try to explain/predict the observations (experimental data), where R is the number
            of predictors and N the number of elements for each predictor. If set to None, the predictors of
            the instance will be used. For this method to work properly the predictors must be the same
            as the ones used in training (fit method).

        prediction_parameters : numpy.array
            array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
            the parameters to fit the predictors to the corrected observations for each variable, where M =
            X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
            variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
            used.

        Returns
        -------

        numpy.array
            The (effective) degrees of freedom for the fitted predictors for each variable
            X1, X2, ..., Xn
        '''
        ## Treat observations
        obs = np.array(observations, dtype=np.float64)
        # Keep original dimensions (to reset dimensions of corrected data)
        dims = obs.shape
        # Make matrix 2-dimensional
        obs = obs.reshape(dims[0], -1)

        # Check correctness of matrix
        if 0 in dims:
            return np.zeros((1, dims[1:]))

        # Treat predictors
        if covariates is None:
            preds = self._crvfitter_covariates
            if 0 in preds.shape:
                raise AttributeError('There are no predictors in this instance')
        else:
            preds = np.array(covariates, dtype=np.float64)
            if len(preds.shape) != 2:
                raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')
            if 0 in preds.shape:
                raise ValueError('There are no elements in argument \'predictors\'')

        # Treat prediction parameters
        if covariates_parameters is None:
            params = self._crvfitter_covariates_parameters
            dims = (1,) + self._crvfitter_dims
        else:
            params = np.array(covariates_parameters, dtype=np.float64)
            # Keep original dimensions (to reset dimensions of prediction)
            dims = params.shape
            # Make matrix 2-dimensional
            params = params.reshape(dims[0], -1)

        if 0 in dims:
            raise ValueError('There are no elements in argument \'prediction_parameters\'')

        df_prediction = self.__df_fitting__(obs, preds, params)

        # Restore original dimensions (except for the first axis)
        return df_prediction.reshape(-1, *dims[1:])

    def __transform__(self, covariates, covariates_parameters, observations, *args, **kwargs):
        '''
        Computes the (effective) degrees of freedom for the fitted predictors.
        This method is not intended to be called outside the CurveFitter class.

        Parameters
        ----------

        observations : numpy.array
            array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
            i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
            explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
            variables and N the number of observations for each variable.

        predictors : numpy.array
            NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
            to try to explain/predict the observations (experimental data), where R is the number of
            predictors and N the number of elements for each predictor.

        prediction_parameters : numpy.array
            (Kr)xM (2-dimensional) matrix, representing the parameters that best fit
            the predictors to the corrected observations for each variable, where M is the number of
            variables and Kr is the number of prediction parameters for each variable.

        Returns
        -------

        x_scores : numpy.array
            NxLxM (3-dimensional) matrix representing the latent factor associated to predictors. N is the number of
            samples, M is the number of observations per sample and L is the latent-space dimension
        y_score : numpy.array
            NxLxM (3-dimensional) matrix representing the latent factor associated to observations. N is the number of
            samples, M is the number of observations per sample and L is the latent-space dimension

        '''
        raise ValueError('No latent subspace is computed with this method. Please, specify the correct fitter.')

    def transform(self, covariates=None, covariates_parameters=None, observations=None, *args, **kwargs):
        '''
        Computes the (effective) degrees of freedom for the fitted predictors

        Parameters
        ----------

        observations : numpy.array
            array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
            i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
            explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
            variables and N the number of observations for each variable. Be aware that the observations
            must be the same as the ones used in prediction, that is, if the prediction was performed over
            corrected observations, these corrected observations are the ones that must be used here.

        predictors : numpy.array
            NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
            to be used to try to explain/predict the observations (experimental data), where R is the number
            of predictors and N the number of elements for each predictor. If set to None, the predictors of
            the instance will be used. For this method to work properly the predictors must be the same
            as the ones used in training (fit method).

        prediction_parameters : numpy.array
            array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
            the parameters to fit the predictors to the corrected observations for each variable, where M =
            X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
            variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
            used.

        Returns
        -------

        x_scores : numpy.array
            NxLxM (3-dimensional) matrix representing the latent factor associated to predictors. N is the number of
            samples, M is the number of observations per sample and L is the latent-space dimension
        y_score : numpy.array
            NxLxM (3-dimensional) matrix representing the latent factor associated to observations. N is the number of
            samples, M is the number of observations per sample and L is the latent-space dimension

        '''

        if observations is None:
            obs=None
        else:
            ## Treat observations
            obs = np.array(observations, dtype=np.float64)
            # Keep original dimensions (to reset dimensions of corrected data)
            dims = obs.shape
            # Make matrix 2-dimensional
            obs = obs.reshape(dims[0], -1)


        # Treat predictors
        if covariates is None:
            preds = self._crvfitter_covariates
            if 0 in preds.shape:
                raise AttributeError('There are no predictors in this instance')
        else:
            preds = np.array(covariates, dtype=np.float64)
            if len(preds.shape) != 2:
                raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')
            if 0 in preds.shape:
                raise ValueError('There are no elements in argument \'predictors\'')

        # Treat prediction parameters
        if covariates_parameters is None:
            params = self._crvfitter_covariates_parameters
            dims = (1,) + self._crvfitter_dims
        else:
            params = np.array(covariates_parameters, dtype=np.float64)
            # Keep original dimensions (to reset dimensions of prediction)
            dims = params.shape
            # Make matrix 2-dimensional
            params = params.reshape(dims[0], -1)

        if 0 in dims:
            raise ValueError('There are no elements in argument \'prediction_parameters\'')

        x_scores, y_scores = self.__transform__(preds, params, obs, *args, **kwargs)

        # Restore original dimensions (except for the first axis)
        return x_scores.reshape(x_scores.shape[:2]+dims[1:]), y_scores.reshape(y_scores.shape[:2] + dims[1:])

    def get_item_parameters(self, parameters, name = None):
        raise ValueError('No get_item_parameters for this fitter')

    @property
    def fitter(self):
        return self


class CombinedFitter(object):

    __threshold = (1e-14 ** 2)

    def __init__(self, correction_fitter, prediction_fitter):
        self._fitter_corrector = correction_fitter
        self._fitter_predictor = prediction_fitter
    #
    #     covariates = correction_fitter
    #
    #     super(self,CurveFitter).__init__(covariates=None, intercept=False)

    def get_item_parameters(self, parameters, name=None):
        return self._fitter_predictor.get_item_parameters(parameters, name)

    def orthogonalize_all(self):
        '''Orthogonalizes each predictor w.r.t the others, all correctors w.r.t. the others, and all the
            predictors w.r.t. all the correctors.

            Modifies:

                - Correctors: each column has been orthogonalized with respect to the previous np.ones.
                - Regressors: each column has been orthogonalized with respect to all the columns in the correctors
                    matrix and all the previous columns in the predictors matrix.

            Returns:

                - Deorthogonalization matrix: A (C+R)x(C+R) (2-dimensional) upper triangular matrix that yields the
                    original 'correctors' and 'predictors' matrices when right-multiplied with the new 'correctors' and
                    'predictors' matrices. More specifically, given the original 'correctors' matrix, OC, the original
                    'predictors' matrix, OR, and the new, orthogonalized 'correctors' and 'predictors' matrices, NC
                    and NR respectively, the return value is a matrix, D, such that (OC | OR) = (NC | NR) x D (matrix
                    multiplication).
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

        C = self._fitter_corrector._crvfitter_correctors.shape[1]
        R = self._fitter_predictor._crvfitter_predictors.shape[1]
        CR = C + R
        D = np.zeros((CR, CR))  # D[i, j] = 0, if i > j

        threshold = self._fitter_corrector._crvfitter_correctors.shape[0] * CombinedFitter.__threshold

        for i in range(C):
            D[i, i] = 1.0  # D[i, j] = 1, if i = j

            u_i = self._fitter_corrector._crvfitter_correctors[:, i]
            norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

            if norm_sq < threshold:
                u_i[
                :] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
                # Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
                # work, hopefully with a small enough precision error)
                continue

            for j in range(i + 1, C):
                v_j = self._fitter_corrector._crvfitter_correctors[:, j]

                D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
                v_j -= D[i, j] * u_i

            for j in range(C, CR):
                v_j = self._fitter_predictor._crvfitter_predictors[:, j - C]

                D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
                v_j -= D[i, j] * u_i

        D[C:, C:] = self._fitter_predictor.orthogonalize_predictors()  # R x R

        return D

    def normalize_all(self):
        '''Normalizes the energy of each corrector and each predictor (the magnitude of each feature
            interpreted as a vector, that is, the magnitude of each column of the internal correctors and
            predictors matrices).

            Modifies:

                - Correctors: each column has been normalized to have unit magnitude.
                - Regressors: each column has been normalized to have unit magnitude.

            Returns:

                - Denormalization matrix: A (C+R)x(C+R) (2-dimensional) diagonal matrix that yields the original
                    'correctors' and 'predictors' matrices when right-multiplied with the new 'correctors' and
                    'predictors' matrices. That is, given the original 'correctors' matrix, namely OC, the original
                    'predictors' matrix, OR, and the new, normalized 'correctors' and 'predictors' matrices, NC and
                    NR respectively, the return value is a diagonal matrix D such that (OC | OR) = (NC | NR) x D
                    (matrix multiplication).
        '''

        # Deorthogonalization matrix (diagonal):
        #    D[i, j] =
        #             ||u_i||,    if i = j
        #             0,            if i != j

        C = self._fitter_corrector._crvfitter_correctors.shape[1]
        R = self._fitter_predictor._crvfitter_predictors.shape[1]
        CR = C + R
        D = np.zeros((CR, CR))

        D[:C, :C] = self._fitter_corrector.normalize_correctors()
        D[C:, C:] = self._fitter_predictor.normalize_predictors()

        return D

    def orthonormalize_all(self):
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

        C = self._fitter_corrector._crvfitter_correctors.shape[1]
        R = self._fitter_predictor._crvfitter_predictors.shape[1]
        CR = C + R
        D = np.zeros((CR, CR))

        threshold = self._fitter_corrector._crvfitter_correctors.shape[0] * CombinedFitter.__threshold

        for i in range(C):
            u_i = self._fitter_corrector._crvfitter_correctors[:, i]

            norm_sq = u_i.dot(u_i)  # < u_i, u_i > = ||u_i||**2
            if norm_sq < threshold:
                u_i[
                :] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
                # Notice that D[i, i] is set to 0, which is exactly the same as ||u_i||, as requested (this means that
                # the deorthonormalization will still work, hopefully with a small enough precision error)
                continue

            D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
            u_i /= D[i, i]  # Normalize u_i, now u_i denotes w_i (step 2 of Gram-Schmidt)

            for j in range(i + 1, C):
                v_j = self._fitter_corrector._crvfitter_correctors[:, j]

                D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
                v_j -= D[
                           i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

            for j in range(C, CR):
                v_j = self._fitter_predictor._crvfitter_predictors[:, j - C]

                D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
                v_j -= D[
                           i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

        D[C:, C:] = self._fitter_predictor.orthonormalize_predictors()  # R x R

        return D

    def fit(self, observations, *args, **kwargs):

        correct_flag = kwargs.get('data_is_corrected', False)
        if not correct_flag:
            # Fit correctors
            self._fitter_corrector.fit(observations, *args, **kwargs)

            # Correct data
            obs = self.correct(observations, correctors=None, correction_parameters=None,*args, **kwargs)

        else:
            obs = observations

        # Fit predictors with corrected data
        self._fitter_predictor.fit(obs, *args, **kwargs)

    def correct(self, observations, correctors=None, correction_parameters=None, *args, **kwargs):
        '''Computes the values of the data after accounting for the correctors by using the correction parameters.
            Parameters:
                - observations: array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
                    i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
                    explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
                    variables and N the number of observations for each variable.
                - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
                    that (may) explain a part of the observational data in which we are not interested, where C is
                    the number of correctors and N the number of elements for each corrector. If set to None, the
                    internal correctors will be used.
                - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
                    the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
                    is the number of variables and Kc the number of correction parameters for each variable. If set
                    to None, the correction parameters obtained in the last call to 'fit' will be used.
                - any other arguments will be passed to the __correct__ method.
            Returns:
                - Corrected data: array-like matrix of shape (N, X1, ..., Xn), containing the observational data
                    after having subtracted the contribution of the correctors by using the correction parameters.
        '''

        ## Treat observations
        obs = np.array(observations, dtype=np.float64)
        # Keep original dimensions (to reset dimensions of corrected data)
        dims = obs.shape
        # Make matrix 2-dimensional
        obs = obs.reshape(dims[0], -1)

        # Check correctness of matrix
        if 0 in dims:
            return np.zeros(dims)

        predicted_data = self._fitter_corrector.__predict__(correctors, correction_parameters, *args, **kwargs)

        if obs.shape != predicted_data.shape:
            raise ValueError('The dimensions of the observations and the predicted data are incompatible. It may indicat'
                             'that either the correctors or correction parameters have wrong shape.')

        return (obs - predicted_data).reshape(dims)

    def predict(self, predictors=None, prediction_parameters=None, *args, **kwargs):
        return self._fitter_predictor.predict(predictors, prediction_parameters, *args, **kwargs)

    def df_correction(self, observations, correctors=None, correction_parameters=None):
        return self._fitter_corrector.df_fitting(observations, correctors, correction_parameters)

    def df_prediction(self, observations, predictors=None, prediction_parameters=None):
        return self._fitter_predictor.df_fitting(observations, predictors, prediction_parameters)

    def evaluate_fit(self, observations, evaluation_function, correctors=None, correction_parameters=None,
                     predictors=None, prediction_parameters=None, *args, **kwargs):

        """Evaluates the degree to which the correctors and predictors get to explain the observational
            data passed in the 'observations' argument by using the evaluator at 'evaluation_function'.

            Parameters:

                - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational data,
                    i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
                    explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
                    variables and N the number of observations for each variable.

                - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
                    that (may) explain a part of the observational data in which we are not interested, where C is
                    the number of correctors and N the number of elements for each corrector. If set to None, the
                    internal correctors will be used.

                - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
                    the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
                    is the number of variables and Kc the number of correction parameters for each variable. If set
                    to None, the correction parameters obtained in the last call to 'fit' will be used.

                - predictors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
                    to be used to try to explain/predict the observations (experimental data), where R is the number
                    of predictors and N the number of elements for each predictor. If set to None, the internal re-
                    gressors will be used.

                - prediction_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
                    the parameters to fit the predictors to the corrected observations for each variable, where M =
                    X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
                    variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
                    used.

                - any other arguments will be passed to the 'evaluation_function.evaluate' method.

            Returns:

                - Fitting scores: array-like structure of shape (X1, ..., Xn), containing floats that indicate the
                    goodness of the fit, that is, how well the predicted curves represent the corrected observational
                    data, or equivalently, how well the model applied to the predictors explains the observed data.
        """

        obs = np.array(observations, dtype=np.float64)
        dims = obs.shape
        obs = obs.reshape(dims[0], -1)

        if 0 in dims:
            raise ValueError('There are no elements in argument \'observations\'')

        if correctors is None:
            cors = self._fitter_corrector.correctors
            if 0 in cors.shape:
                correctors_present = False
            else:
                correctors_present = True
        else:
            cors = np.array(correctors, dtype=np.float64)
            if len(cors.shape) != 2:
                raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')

            if 0 in cors.shape:
                raise ValueError('There are no elements in argument \'correctors\'')

            correctors_present = True

        if correctors_present:
            if obs.shape[0] != cors.shape[0]:
                raise ValueError('The dimensions of the observations and the correctors are incompatible')

            if correction_parameters is None:
                cparams = self._fitter_corrector.correction_parameters
                if 0 in cparams.shape:
                    raise AttributeError('There are no correction parameters in this instance')
            else:
                cparams = np.array(correction_parameters, dtype=np.float64)
                cparams = cparams.reshape(cparams.shape[0], -1)

                if 0 in cparams.shape:
                    raise ValueError('There are no elements in argument \'correction_parameters\'')

            if obs.shape[1] != cparams.shape[1]:
                raise ValueError('The dimensions of the observations and the correction parameters are incompatible')

        else:
            cparams = np.zeros((0, 0))

        if predictors is None:
            preds = self._fitter_predictor.predictors
            if 0 in preds.shape:
                raise AttributeError('There are no predictors in this instance')
        else:
            preds = np.array(predictors, dtype=np.float64)

            if len(preds.shape) != 2:
                raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')

            if 0 in preds.shape:
                raise ValueError('There are no elements in argument \'predictors\'')

        if obs.shape[0] != preds.shape[0]:
            raise ValueError('The dimensions of the observations and the predictors are incompatible')

        if prediction_parameters is None:
            pparams = self._fitter_predictor.prediction_parameters
            if 0 in pparams.shape:
                raise AttributeError('There are no prediction parameters in this instance')
        else:
            pparams = np.array(prediction_parameters, dtype=np.float64)
            # Make matrix 2-dimensional
            pparams = pparams.reshape(pparams.shape[0], -1)

            if 0 in pparams.shape:
                raise ValueError('There are no elements in argument \'prediction_parameters\'')

        if obs.shape[1] != pparams.shape[1]:
            raise ValueError('The dimensions of the observations and the prediction parameters are incompatible')

        class FittingResults(object):
            pass

        fitres = FittingResults()

        fitres.observations = obs
        fitres.correctors = cors
        fitres.correction_parameters = cparams
        fitres.predictors = preds
        fitres.prediction_parameters = pparams

        fitting_scores = evaluation_function[self].evaluate(fitres, *args, **kwargs)

        return fitting_scores.reshape(dims[1:])


    def transform(self, predictors=None, prediction_parameters=None, corrected_observations=None, *args, **kwargs):
        return self._fitter_predictor.transform(predictors, prediction_parameters, corrected_observations, *args, **kwargs)

    @property
    def correction_parameters(self):
        return self._fitter_corrector.correction_parameters

    @property
    def prediction_parameters(self):
        return self._fitter_predictor.prediction_parameters

    @property
    def correctors(self):
        return self._fitter_corrector.correctors

    @property
    def predictors(self):
        return self._fitter_predictor.predictors

    @property
    def prediction_fitter(self):
        return self._fitter_predictor



class NullFitter(CurveFitter):

    @staticmethod
    def __predict__(predictors, prediction_parameters, *args, **kwargs):
        return np.zeros((predictors.shape[0],prediction_parameters.shape[1]))

    @staticmethod
    def __correct__(observations, correctors, correction_parameters, *args, **kwargs):
        return observations

    @staticmethod
    def __fit__(correctors, predictors, observations, *args, **kwargs):
        pass



""" FIT EVALUATION BINDINGS """

eval_func[CurveFitter].bind(
    'corrected_data',
    lambda self: self.target.correct(
        observations=self.fitting_results.observations,
        correctors=self.fitting_results.correctors,
        correction_parameters=self.fitting_results.correction_parameters
    )
)
eval_func[CurveFitter].bind(
    'predicted_data',
    lambda self: self.target.predict(
        predictors=self.fitting_results.predictors,
        prediction_parameters=self.fitting_results.prediction_parameters
    )
)
eval_func[CurveFitter].bind(
    'df_correction',
    lambda self: self.target.df_correction(
        observations=self.fitting_results.observations,
        correctors=self.fitting_results.correctors,
        correction_parameters=self.fitting_results.correction_parameters
    )
)
eval_func[CurveFitter].bind(
    'df_prediction',
    lambda self: self.target.df_prediction(
        observations=self.fitting_results.observations,
        predictors=self.fitting_results.predictors,
        prediction_parameters=self.fitting_results.prediction_parameters
    )
)
