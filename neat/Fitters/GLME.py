import numpy as np
from sklearn.linear_model import LinearRegression as LR

from neat.Fitters.CurveFitting import CurveFitter
from neat.Fitters.GLM import PolyGLM

from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams

class CurveFitterLongitudinal(CurveFitter):
    def __init__(self, predictors=None, correctors=None, correctors_random_effects=None, groups=None,
                 intercept_fe=CurveFitter.NoIntercept, intercept_re = True):
        '''[Constructor]

            Parameters:

                - predictors: NxR (2-dimensional) matrix (default None), representing the predictors,
                    i.e., features to be used to try to explain/predict some observations (experimental
                    data), where R is the number of predictors and N the number of elements for each
                    predictor.

                - correctors: NxC (2-dimensional) matrix (default None), representing the covariates,
                    i.e., features that (may) explain a part of the observational data in which we are
                    not interested, where C is the number of correctors and N the number of elements
                    for each corrector (the latter must be the same as that in the 'predictors' argu-
                    ment).

                - intercept: one of CurveFitter.NoIntercept, CurveFitter.PredictionIntercept or
                    CurveFitter.CorrectionIntercept (default CurveFitter.NoIntercept), indicating whether
                    the intercept (a.k.a. homogeneous term) must be incorporated to the model or not, and
                    if so, wheter it must be as a predictor or a corrector. In the last two cases, a column
                    of ones will be added as the first column (feature) of the internal predictors/correctors
                    matrix. Please notice that, if the matrix to which the columns of ones must precede
                    does not have any elements, then this parameter will have no effect.
        '''

        if not predictors is None:
            predictors = np.array(predictors, dtype=np.float64)

            if len(predictors.shape) != 2:
                raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')

        if not correctors is None:
            correctors = np.array(correctors, dtype=np.float64)

            if len(correctors.shape) != 2:
                raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix (or None)')

        if not correctors_random_effects is None:
            correctors_random_effects = np.array(correctors_random_effects, dtype=np.float64)

            if len(correctors_random_effects.shape) != 2:
                raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix (or None)')


        if predictors is None:
            if correctors is None:
                self._crvfitter_correctors = np.zeros((0, 0))
                self._crvfitter_predictors = np.zeros((0, 0))
            else:
                if intercept_fe == CurveFitter.PredictionIntercept:
                    self._crvfitter_correctors = correctors
                    self._crvfitter_predictors = np.ones((self._crvfitter_correctors.shape[0], 1))
                else:
                    self._crvfitter_predictors = np.zeros((correctors.shape[0], 0))
                    if intercept_fe == CurveFitter.CorrectionIntercept:
                        self._crvfitter_correctors = np.concatenate((np.ones((correctors.shape[0], 1)), correctors),
                                                                    axis=1)
                    else:
                        self._crvfitter_correctors = correctors

        else:
            if correctors is None:
                if intercept_fe == CurveFitter.CorrectionIntercept:
                    self._crvfitter_predictors = predictors
                    self._crvfitter_correctors = np.ones((self._crvfitter_predictors.shape[0], 1))
                else:
                    self._crvfitter_correctors = np.zeros((predictors.shape[0], 0))
                    if intercept_fe == CurveFitter.PredictionIntercept:
                        self._crvfitter_predictors = np.concatenate((np.ones((predictors.shape[0], 1)), predictors),
                                                                    axis=1)
                    else:
                        self._crvfitter_predictors = predictors
            else:
                if correctors.shape[0] != predictors.shape[0]:
                    raise ValueError(
                        'Correctors and predictors must have the same number of samples (same length in the first dimension)')

                if intercept_fe == CurveFitter.CorrectionIntercept:
                    self._crvfitter_correctors = np.concatenate((np.ones((correctors.shape[0], 1)), correctors), axis=1)
                    self._crvfitter_predictors = predictors
                elif intercept_fe == CurveFitter.PredictionIntercept:
                    self._crvfitter_predictors = np.concatenate((np.ones((predictors.shape[0], 1)), predictors), axis=1)
                    self._crvfitter_correctors = correctors
                else:
                    self._crvfitter_correctors = correctors
                    self._crvfitter_predictors = predictors

        if correctors_random_effects is None:
            if intercept_re is True:
                self._crvfitter_correctors_re = np.ones((self._crvfitter_correctors.shape[0], 1))
        else:
            if intercept_re is True:
                self._crvfitter_correctors_re = np.concatenate((np.ones((self._crvfitter_correctors.shape[0], 1)),
                                                                correctors_random_effects),axis=1)
            else:
                self._crvfitter_correctors_re = correctors_random_effects


        self.groups = groups

        C = self._crvfitter_correctors.shape[1]
        R = self._crvfitter_predictors.shape[1]
        self._crvfitter_correction_parameters = np.zeros((C, 0))
        self._crvfitter_prediction_parameters = np.zeros((R, 0))

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

        self._crvfitter_dims = dims[1:]
        if dims[0] != self._crvfitter_predictors.shape[0]:
            raise ValueError('Observations and features (correctors and/or predictors) have incompatible sizes')

        if 0 in dims:
            raise ValueError('There are no elements in argument \'observations\'')

        obs = obs.reshape(dims[0], -1)
        self._crvfitter_correction_parameters, self._crvfitter_prediction_parameters = self.__fit__(
            self._crvfitter_correctors, self._crvfitter_correctors_re, self.groups,
            self._crvfitter_predictors,  obs, *args, **kwargs)
        return self



class CurveFitterLongitudinal(PolyGLM):
    ''' Class that implements the General Linear Mixed Effects Model.

        This method assumes the following situation:

            - There are M (random) variables whose behaviour we want to explain.

            - Each of the M variables has been measured N times, obtaining thus
              an NxM matrix of observations (i-th column contains the N observa-
              tions for i-th variable).

            - There are K predictors (in this class both, the correctors and the
              predictors are called predictors and treated equally) that might
              explain the behaviour of the M variables in an additive manner, i.e.,
              a ponderated sum of the K predictors might fit each of the variables.
              Fixed-effects

            - Each of the K predictors has been measured at the same moments in
              which the M variables were measured, giving thus a NxK matrix where
              the i-th column represents the N observations of the i-th predictor.

            - There are P different predictors (in this class both, the correctors
              and the predictors are called predictors and treated equally) that might
              explain the behaviour of the M variables in an additive manner, i.e.,
              a ponderated sum of the K predictors might fit each of the variables.
              Each of these P predictors are embedded into a matrix whose strucure
              is Nx(L·P) where L is the number of dependent observations (i.e. multiple
              observations of the same subject). Random-effects


        
        In this situation, the relationship of the different elements can be ex-
        pressed as follows:

            OBS(NxM) = MODEL1(NxK) * PARAMS(KxM) + MODEL2(NxLP) * PARAMS_RE(LPxM) +eps(NxM),

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
    def __fit__(correctors, correctors_re, groups, predictors, observations, sample_weight=None, n_jobs=-1,
                *args, **kwargs):

        ncols = correctors.shape[1]
        dims = (correctors.shape[0], ncols + predictors.shape[1])
        xdata = np.zeros(dims)
        xdata[:, :ncols] = correctors.view()
        xdata[:, ncols:] = predictors.view()

        M = observations.shape[1]
        K = correctors.shape[1]


        params = np.empty((K,M), dtype=object)
        for it_m in range(M):
            free = MixedLMParams.from_components(fe_params = np.ones(xdata.shape[1]),
                                                 cov_re = np.eye(correctors_re.shape[1]))
            model = MixedLM(
                endog = observations,
                exog = xdata,
                groups = groups,
                exog_re = correctors_re
            )

            results = model.fit(free=free)
            params[..., it_m] = free

        return (params[:ncols], params[ncols:])



    def __df_correction__(self, observations, correctors, correction_parameters):
        print('Dof for Longitudinal Random Effects modeling is still not available. Instead, using the standard GLM df')
        return super(GLME,self).__df_correction__(observations, correctors, correction_parameters)

    def __df_prediction__(self, observations, predictors, prediction_parameters):
        print('Dof for Longitudinal Random Effects modeling is still not available. Instead, using the standard GLM df')
        return super(GLME,self).__df_prediction__(observations, predictors, prediction_parameters)

