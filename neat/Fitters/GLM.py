import numpy as np
from sklearn.linear_model import LinearRegression as LR

from neat.Fitters.CurveFitting import CurveFitter
from neat.Utils.Transforms import polynomial


class GLM(CurveFitter):
    ''' Class that implements the General Linear Method.

        This method assumes the following situation:

            - There are M (random) variables whose behaviour we want to explain.

            - Each of the M variables has been measured N times, obtaining thus
              an NxM matrix of observations (i-th column contains the N observa-
              tions for i-th variable).

            - There are K predictors (in this class both, the correctors and the
              predictors are called predictors and treated equally) that might
              explain the behaviour of the M variables in an additive manner, i.e.,
              a ponderated sum of the K predictors might fit each of the variables.

            - Each of the K predictors has been measured at the same moments in
              which the M variables were measured, giving thus a NxK matrix where
              the i-th column represents the N observations of the i-th predictor.
        
        In this situation, the relationship of the different elements can be ex-
        pressed as follows:

            OBS(NxM) = MODEL(NxK) * PARAMS(KxM) + eps(NxM),

        where OBS denotes the NxM matrix containing the N observations of each of
        the M variables, MODEL denotes the NxK matrix containing the N observations
        of each of the K predictors, PARAMS denotes the KxM matrix of ponderation
        coefficients (one for each variable and predictor, that is, the amplitude
        each predictor has in each variable), and eps denotes the error commited
        when making the aforementioned assumptions, i.e., a NxM matrix that contains
        the data that is left unexplained after accounting for all the predictors
        in the model.

        This class provides the tools to orthogonalize each of the predictors in
        the matrix with respect to the ones in the previous columns, and to esti-
        mate the ponderation coefficients (the PARAMS matrix) so that the energy
        of the error (the MSE) is minimized.
    '''

    @staticmethod
    def __predict__(covariates, covariate_parameters, *args, **kwargs):
        '''Computes a prediction applying the prediction function used in GLM.

            Parameters:

                - predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
                    to try to explain/predict the observations (experimental data), where R is the number of
                    predictors and N the number of elements for each predictor.

                - prediction_parameters: RxM (2-dimensional) matrix, representing the parameters that best fit
                    the predictors to the corrected observations for each variable, where M is the number of
                    variables and K is the number of prediction parameters for each variable.

                - any other arguments will also be passed to the method in the subclass.

            Returns:

                - Prediction: NxM (2-dimensional) matrix, containing N predicted values for each of the M variables,
                    result of computing the expression 'predictors * prediction_parameters' (matrix multiplication).
        '''
        return covariates.dot(covariate_parameters)

    @staticmethod
    def __fit__(covariates, observations, sample_weight=None, n_jobs=-1, *args, **kwargs):
        '''Computes the correction and prediction parameters that best fit the observations according to the
            General Linear Model.

            Parameters:

                - covariates: NxC (2-dimensional) matrix, representing the covariates, i.e., features that
                    (may) explain a part of the observational data in which we are not interested, where C
                    is the number of correctors and N the number of elements for each corrector.

                - observations: NxM (2-dimensional) matrix, representing the observational data, i.e., values
                    obtained by measuring the variables of interest, whose behaviour is wanted to be explained
                    by the correctors and predictors, where M is the number of variables and N the number of
                    observations for each variable (the latter is ensured to be the same as those in the
                    'correctors' and the 'predictors' arguments).

                - sample_weight: array of length N (default None), indicating the weight of each sample for
                    the fitting algorithm, where N is the number of observations for each variable. If set
                    to None, each sample will have the same weight.

                - num_threads: integer (default -1), indicating the number of threads to be used by the algo-
                    rithm. If set to -1, all CPUs are used. This will only provide speed-up for M > 1 and
                    sufficiently large problems.

            Returns:

                - Regression parameters: RxM (2-dimensional) matrix, representing the parameters that best fit
                    the predictors to the corrected observations for each variable, where M is the number of
                    variables (same as that in the 'observations' argument) and R is the number of prediction
                    parameters for each variable (same as the number of predictors).
        '''
        # All-at-once approach

        curve = LR(fit_intercept=False, normalize=False, copy_X=False, n_jobs=n_jobs)

        xdata = covariates.view()

        curve.fit(xdata, observations, sample_weight)
        params = curve.coef_.T
        return params

    def __df_fitting__(self, observations, covariates, covariate_parameters):
        return np.ones((1, observations.shape[1])) * covariates.shape[1]

    def num_estimated_parameters(self, covariate_parameters):
        return covariate_parameters.shape[0] + 1

    def max_loglikelihood_value(self, observations, covariates, covariate_parameters):
        # Compute residuals
        # corrected_values = self.correct(observations, correctors, correction_parameters)
        residuals = observations - self.predict(covariates, covariate_parameters)

        # Compute residual sum of squares
        rss = np.sum(np.square(residuals), axis=0)

        # Number of samples N
        n = observations.shape[0]

        return (-n / 2) * (np.log(2 * np.pi) + np.log(rss) - np.log(n) + 1)


class PolyGLM(GLM):
    def __init__(self, features, degrees=None, intercept=CurveFitter.IncludeIntercept):
        '''[Constructor]

            Parameters:

                - features: NxF (2-dimensional) matrix representing the features to be used to try to
                    explain some observations (experimental data), either by using them as correctors/
                    covariates or predictors/predictors in the model, where F is the number of features
                    and N the number of samples for each feature.

                - predictors: int / iterable object (default None), containing the index/indices of the
                    column(s) in the 'features' matrix that must be used as predictors. If set to None,
                    all the columns of such matrix will be interpreted as predictors.

                - degrees: iterable of F elements (default None), containing the degree of each feature
                    in the 'features' argument, where F is the number of features. If set to None, only
                    the linear term of each feature will be taken into account (same as setting all the
                    degrees to 1).

                - intercept: one of PolyGLM.NoIntercept, PolyGLM.PredictionIntercept or
                    PolyGLM.CorrectionIntercept (default PolyGLM.NoIntercept), indicating whether
                    the intercept (a.k.a. homogeneous term) must be incorporated to the model or not, and
                    if so, wheter it must be as a predictor or a corrector. In the last two cases, a column
                    of ones will be added as the first column (feature) of the internal predictors/correctors
                    matrix. Please notice that, if the matrix to which the columns of ones must precede
                    does not have any elements, then this parameter will have no effect.
        '''
        self._pglm_features = np.array(features)
        if len(self._pglm_features.shape) != 2:
            raise ValueError('Argument \'features\' must be a 2-dimensional matrix')
        self._pglm_features = self._pglm_features.T
        self._pglm_is_predictor = [True] * len(self._pglm_features)

        # if predictors is None:
        #     self._pglm_is_predictor = [True] * len(self._pglm_features)
        #     predictors = []
        # else:
        #     self._pglm_is_predictor = [False] * len(self._pglm_features)
        #     if isinstance(predictors, int):
        #         predictors = [predictors]
        #
        # try:
        #     for r in predictors:
        #         try:
        #             self._pglm_is_predictor[r] = True
        #         except TypeError:
        #             raise ValueError('All elements in argument \'predictors\' must be valid indices')
        #         except IndexError:
        #             raise IndexError('Index out of range in argument \'predictors\'')
        # except TypeError:
        #     raise TypeError('Argument \'predictors\' must be iterable or int')

        if degrees is None:
            self._pglm_degrees = [1] * len(self._pglm_features)
        else:
            degrees = list(degrees)
            if len(degrees) != len(self._pglm_features):
                raise ValueError('Argument \'degrees\' must have a length equal to the number of features')
            for deg in degrees:
                if not isinstance(deg, int):
                    raise ValueError('Expected integer in \'degrees\' list, got ' + str(type(deg)) + ' instead')
                if deg < 1:
                    raise ValueError('All degrees must be >= 1')
            self._pglm_degrees = degrees

        self._pglm_intercept = intercept

        self.__pglm_update_GLM()

    def __pglm_update_GLM(self):
        '''Private function. Not meant to be used by anyone outside the PolyGLM class.
        '''
        covariates = []
        for index in range(len(self._pglm_is_predictor)):
            for p in polynomial(self._pglm_degrees[index], [self._pglm_features[index]]):
                covariates.append(p)

        if len(covariates) == 0:
            covariates = None
        else:
            covariates = np.array(covariates).T

        super(PolyGLM, self).__init__(covariates, self._pglm_intercept)

    @property
    def lin_covariates(self):
        '''Matrix containing the linear terms of the features that are interpreted as predictors in the model.
        '''
        r = [self._pglm_features[i] for i in range(len(self._pglm_is_predictor)) if self._pglm_is_predictor[i]]
        return np.array(r).copy().T

    def set_covariates(self, covariates):
        '''Reselects the predictors of the model.

            Parameters:

                - covariates: int / iterable object (default None), containing the index/indices of the
                    column(s) in the 'features' matrix that must be used as predictors. If set to None,
                    all the columns of such matrix will be interpreted as predictors.

            Modifies:

                - Covariates: the new covariates are set, deorthogonalized and denormalized.

                - [deleted] Correction parameters

                - [deleted] Regression parameters
        '''
        if covariates is None:
            pglm_is_predictor = [True] * len(self._pglm_features)
            predictors = []
        else:
            pglm_is_predictor = [False] * len(self._pglm_features)
            if isinstance(covariates, int):
                predictors = [covariates]

        try:
            for r in predictors:
                try:
                    pglm_is_predictor[r] = True
                except TypeError:
                    raise ValueError('All elements in argument \'predictors\' must be valid indices')
                except IndexError:
                    raise IndexError('Index out of range in argument \'predictors\'')
        except TypeError:
            raise TypeError('Argument \'predictors\' must be iterable or int')

        self._pglm_is_predictor = pglm_is_predictor

        self.__pglm_update_GLM()
