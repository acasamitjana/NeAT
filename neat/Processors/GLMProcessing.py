import numpy as np

from neat.Fitters.GLM import GLM, PolyGLM as PGLM
from neat.Processors.Processing import Processor


class GLMProcessor(Processor):
    _glmprocessor_perp_norm_options_names = [
        'Orthonormalize covariates',
        'Orthogonalize covariates',
        'Normalize covariates',
        'Use covariates as they are'
    ]

    _glmprocessor_perp_norm_options_list = [
        GLM.orthonormalize_covariates,
        GLM.orthogonalize_covariates,
        GLM.normalize_covariates,
        lambda *args, **kwargs: np.zeros((0, 0))
    ]

    _glmprocessor_intercept_options_names = [
        'Do not include the intercept term',
        'Include the intercept term'
    ]

    _glmprocessor_intercept_options_list = [
        False,
        True
    ]

    _glmprocessor_submodels_options_names = [
        'Do not include this term in the system',
        'As a corrector',
        'As a predictor'
    ]

    def _glmprocessor_compute_original_parameters(self, Gamma, Beta2):
        '''Given an upper triangular matrix Gamma, and an arbitrary matrix Beta2, computes Beta such
            that Beta2 = Gamma * Beta.
            Notice that if Beta2 is the np.identity matrix, then Beta is the right-pseudoinverse of Gamma.
        '''

        # Gamma is the deorthogonalization (upper triangular) matrix
        # Beta2 is the matrix with the optimal parameters of the orthonormalized design matrix
        # Beta is the matrix with the optimal parameters of the original design matrix

        # The relationships between the different matrices of the system are described below:
        # (1) Y = X * Beta
        # (2) X = Z * Gamma
        # (3) Y = Z * Beta2

        # dim(Y) = NxM
        # dim(X) = dim(Z) = NxK
        # dim(Beta) = dim(Beta2) = KxM
        # dim(Gamma) = KxK

        # Combining the three expressions, we get:
        # Z * Beta2 = Z * Gamma * Beta

        # Thus, we can get the elements of Beta by solving the following equation:
        # Beta2 = Gamma * Beta

        # where Gamma is an upper triangular matrix (look out, it could be singular, which is why
        # we do not apply Beta = inv(Gamma)*Beta2).


        # However, the 'Gamma' argument of this method is actually only a part of the 'Gamma' matrix described
        # above, so we only have to adjust the corresponding part of the 'Beta2' matrix (the rest will be left
        # equal in the 'Beta' matrix).

        Beta = Beta2.copy()
        K = Gamma.shape[0]

        # Get the part of Beta2 that must be 'deorthonormalized'
        if self._glmprocessor_perp_norm_option < 3:
            # All features were orthonormalized/orthogonalized/normalized
            # Process the whole matrix
            dnBeta2 = Beta2.view()
            dnBeta = Beta.view()
        elif self._glmprocessor_perp_norm_option < 6:
            # Only the predictors were orthonormalized/orthogonalized/normalized
            # Only process the last K parameters (the ones belonging to the predictors)
            dnBeta2 = Beta2[-K:].view()
            dnBeta = Beta[-K:].view()
            # Leave the rest as is
            Beta[:-K] = Beta2[:-K].view()
        elif self._glmprocessor_perp_norm_option < 9:
            # Only the correctors were orthonormalized/orthogonalized/normalized
            # Only process the first K parameters (the ones belonging to the correctors)
            dnBeta2 = Beta2[:K].view()
            dnBeta = Beta[:K].view()
            # Leave the rest as is
            Beta[K:] = Beta2[K:].view()
        else:
            # Nothing changed; Beta2 already contains the non-orthogonalized parameters, as does Beta (copied)
            return Beta

        # Work with dnBeta, dnBeta2, and Gamma
        for index in range(K):
            j = K - index - 1
            if Gamma[j, j] == 0:
                continue
            dnBeta[j] /= Gamma[j, j]
            for i in range(j):
                dnBeta[i] -= dnBeta[j] * Gamma[i, j]

        return Beta

    def __fitter__(self, user_defined_parameters):
        """
        Initializes the GLM fitter to be used to process the data.
        """

        # preds = self.predictors.T
        covs = self.covariates.T
        num_features = covs.shape[0]

        self._glmprocessor_intercept = user_defined_parameters[0]
        self._glmprocessor_perp_norm_option = user_defined_parameters[1]
        self._glmprocessor_degrees = user_defined_parameters[2:(2 + num_features)]
        # self._glmprocessor_submodels = user_defined_parameters[(2 + num_features):]

        treat_data = GLMProcessor._glmprocessor_perp_norm_options_list[self._glmprocessor_perp_norm_option]
        intercept = GLMProcessor._glmprocessor_intercept_options_list[self._glmprocessor_intercept]

        covariates = []
        for i in range(len(covs)):
            cor = 1
            for _ in range(self._glmprocessor_degrees[i]):
                cor *= covs[i]
                covariates.append(cor.copy())

        # predictors = []
        # j = 0
        # for i in range(len(preds)):
        #     reg = 1
        #     for _ in range(self._glmprocessor_degrees[i]):
        #         reg *= preds[i]
        #         if self._glmprocessor_submodels[j] == 2:
        #             predictors.append(reg.copy())
        #         elif self._glmprocessor_submodels[j] == 1:
        #             correctors.append(reg.copy())
        #         j += 1
        #
        # correctors = np.array(correctors).T
        # if 0 in correctors.shape:
        #     correctors = None

        if len(covariates) == 0:
            covariates = None
        else:
            covariates = np.atleast_2d(covariates).T

        self._glmprocessor_glm = GLM(covariates=covariates, intercept=intercept)
        self._glmprocessor_deorthonormalization_matrix = treat_data(self._glmprocessor_glm)
        return self._glmprocessor_glm


    def __user_defined_parameters__(self, fitter):
        return (self._glmprocessor_intercept, self._glmprocessor_perp_norm_option) + tuple(
            self._glmprocessor_degrees)# + tuple(self._glmprocessor_submodels)

    def __read_user_defined_parameters__(self, covariates_names, perp_norm_option_global=False,
                                         *args, **kwargs):
        # Intercept term
        # If there are no predictor names, show only options NoIntercept and CorrectionIntercept,
        # and if there are no corrector names, show only NoIntercept and PredictionIntercept. Otherwise,
        # show all options

        default_value = GLMProcessor._glmprocessor_intercept_options_names[1]
        options_names = GLMProcessor._glmprocessor_intercept_options_names

        intercept = GLMProcessor._glmprocessor_intercept_options[super(GLMProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='GLM Processor: How do you want to include the intercept term? (default: {})'.format(
                default_value
            )
        )]

        default_value = GLMProcessor._glmprocessor_perp_norm_options_names[0]
        options_names = GLMProcessor._glmprocessor_perp_norm_options_names


        perp_norm_option = GLMProcessor._glmprocessor_perp_norm_options[super(GLMProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='GLM Processor: How do you want to treat the features? (default: ' +
                      default_value + ')'
        )]



        degrees = []
        for reg in covariates_names:
            degrees.append(super(GLMProcessor, self).__getint__(
                default_value=1,
                lower_limit=1,
                try_ntimes=3,
                show_text='GLM Processor: Please, enter the degree of the feature  \'' + str(
                    reg) + '\' (or leave blank to set to 1): '
            ))
        # for cor in corrector_names:
        #     degrees.append(super(GLMProcessor, self).__getint__(
        #         default_value=1,
        #         try_ntimes=3,
        #         show_text='GLM Processor: Please, enter the degree of the feature \'' + str(
        #             cor) + '\' (or leave blank to set to 1): '
        #     ))

        # submodels = []
        # for i in range(len(predictor_names)):
        #     reg = predictor_names[i]
        #     submodels_text = 'GLM Processor: Would you like to analyze a submodel of {} instead of the full model? ' \
        #                      '(Y/N, default N): '.format(reg)
        #     if super(GLMProcessor, self).__getyesorno__(default_value=False,
        #                                                 show_text=submodels_text):
        #         for j in range(degrees[i]):
        #             submodels.append(
        #                 GLMProcessor._glmprocessor_submodels_options[super(GLMProcessor, self).__getoneof__(
        #                     GLMProcessor._glmprocessor_submodels_options_names,
        #                     default_value=GLMProcessor._glmprocessor_submodels_options_names[2],
        #                     show_text='How should the power ' + str(
        #                         j + 1) + ' term be included in the system? (default: ' +
        #                               GLMProcessor._glmprocessor_submodels_options_names[2] + ')'
        #                 )])
        #     else:
        #         submodels += [2] * degrees[i]

        return (intercept, perp_norm_option) + tuple(degrees)# + tuple(submodels)

    def __curve__(self, fitter, covariate, covariate_parameters, *args, **kwargs):

        # Generate all the necessary terms of the predictor
        preds = covariate.T
        predictors = preds

        # predictors = []
        # j = 0
        # for i in range(len(preds)):
        #     reg = 1
        #     for _ in range(self._glmprocessor_degrees[i]):
        #         reg *= preds[i]
        #         if self._glmprocessor_submodels[j] == 2:
        #             predictors.append(reg.copy())
        #         j += 1


        # Initialize the glm with such predictors
        glm = GLM(covariates=np.array(predictors).T,
                  intercept=GLMProcessor._glmprocessor_intercept_options_list[self._glmprocessor_intercept])
        treat_data = GLMProcessor._glmprocessor_perp_norm_options_list[self._glmprocessor_perp_norm_option]
        treat_data(glm)
        # Get the prediction parameters for the original features matrix


        # Call the normal function with such parameters
        return glm.predict(covariate_parameters=covariate_parameters)

    def __assign_bound_data__(self, observations, covariates, covariate_parameters, fitting_results):
        # Pre-process parameters for fitter operations (predict, correct, etc.) and leave original
        # parameters for processor operations (curve)
        processed_prediction_parameters, processed_correction_parameters = self.__pre_process__(
            covariate_parameters,
            covariates
        )
        # Assign data to compute AIC
        fitting_results.num_estimated_parameters = self._processor_fitter.num_estimated_parameters(
            covariate_parameters=covariate_parameters,
        )
        fitting_results.max_loglikelihood_value = self._processor_fitter.max_loglikelihood_value(
            observations=observations,
            covariates=covariates,
            covariate_parameters=processed_prediction_parameters
        )
        bound_functions = ['num_estimated_parameters', 'max_loglikelihood_value']
        # Call parent method
        bound_functions += super(GLMProcessor, self).__assign_bound_data__(observations, covariates,
                                                                           covariate_parameters, fitting_results)
        return bound_functions

    def get_name(self):
        return 'GLM'


GLMProcessor._glmprocessor_perp_norm_options = {GLMProcessor._glmprocessor_perp_norm_options_names[i]: i for i in
                                                range(len(GLMProcessor._glmprocessor_perp_norm_options_names))}
GLMProcessor._glmprocessor_intercept_options = {GLMProcessor._glmprocessor_intercept_options_names[i]: i for i in
                                                range(len(GLMProcessor._glmprocessor_intercept_options_names))}
GLMProcessor._glmprocessor_submodels_options = {GLMProcessor._glmprocessor_submodels_options_names[i]: i for i in
                                                range(len(GLMProcessor._glmprocessor_submodels_options_names))}


class PolyGLMProcessor(Processor):
    _pglmprocessor_perp_norm_options_names = [
        'Orthonormalize covariates',
        'Orthogonalize covariates',
        'Normalize covariates',
        'Use covariates as they are'
    ]

    _pglmprocessor_perp_norm_options_list = [
        GLM.orthonormalize_covariates,
        GLM.orthogonalize_covariates,
        GLM.normalize_covariates,
        lambda *args, **kwargs: np.zeros((0, 0))
    ]

    _pglmprocessor_intercept_options_names = [
        'Do not include the intercept term',
        'Include the intercept term'
    ]

    _pglmprocessor_intercept_options_list = [
        False,
        True
    ]



    def _pglmprocessor_compute_original_parameters(self, Gamma, Beta2):
        """
        Given an upper triangular matrix Gamma, and an arbitrary matrix Beta2, computes Beta such
        that Beta2 = Gamma * Beta.
        Notice that if Beta2 is the np.identity matrix, then Beta is the right-pseudoinverse of Gamma.
        """

        # Gamma is the deorthogonalization (upper triangular) matrix
        # Beta2 is the matrix with the optimal parameters of the orthonormalized design matrix
        # Beta is the matrix with the optimal parameters of the original design matrix

        # The relationships between the different matrices of the system are described below:
        # (1) Y = X * Beta
        # (2) X = Z * Gamma
        # (3) Y = Z * Beta2

        # dim(Y) = NxM
        # dim(X) = dim(Z) = NxK
        # dim(Beta) = dim(Beta2) = KxM
        # dim(Gamma) = KxK

        # Combining the three expressions, we get:
        # Z * Beta2 = Z * Gamma * Beta

        # Thus, we can get the elements of Beta by solving the following equation:
        # Beta2 = Gamma * Beta

        # where Gamma is an upper triangular matrix (look out, it could be singular, which is why
        # we do not apply Beta = inv(Gamma)*Beta2).


        # However, the 'Gamma' argument of this method is actually only a part of the 'Gamma' matrix described
        # above, so we only have to adjust the corresponding part of the 'Beta2' matrix (the rest will be left
        # equal in the 'Beta' matrix).

        Beta = Beta2.copy()
        K = Gamma.shape[0]

        # Get the part of Beta2 that must be 'deorthonormalized'
        if self._pglmprocessor_perp_norm_option < 3:
            # All features were orthonormalized/orthogonalized/normalized
            # Process the whole matrix
            dnBeta2 = Beta2.view()
            dnBeta = Beta.view()
        elif self._pglmprocessor_perp_norm_option < 6:
            # Only the predictors were orthonormalized/orthogonalized/normalized
            # Only process the last K parameters (the ones belonging to the predictors)
            dnBeta2 = Beta2[-K:].view()
            dnBeta = Beta[-K:].view()
            # Leave the rest as is
            Beta[:-K] = Beta2[:-K].view()
        elif self._pglmprocessor_perp_norm_option < 9:
            # Only the correctors were orthonormalized/orthogonalized/normalized
            # Only process the first K parameters (the ones belonging to the correctors)
            dnBeta2 = Beta2[:K].view()
            dnBeta = Beta[:K].view()
            # Leave the rest as is
            Beta[K:] = Beta2[K:].view()
        else:
            # Nothing changed; Beta2 already contains the non-orthogonalized parameters, as does Beta (copied)
            return Beta

        # Work with dnBeta, dnBeta2, and Gamma
        for index in range(K):
            j = K - index - 1
            if Gamma[j, j] == 0:
                continue
            dnBeta[j] /= Gamma[j, j]
            for i in range(j):
                dnBeta[i] -= dnBeta[j] * Gamma[i, j]

        return Beta

    def __fitter__(self, user_defined_parameters):
        '''Initializes the PolyGLM fitter to be used to process the data.


        '''
        self._pglmprocessor_intercept = user_defined_parameters[0]
        self._pglmprocessor_perp_norm_option = user_defined_parameters[1]
        self._pglmprocessor_degrees = user_defined_parameters[2:]

        treat_data = PolyGLMProcessor._pglmprocessor_perp_norm_options_list[self._pglmprocessor_perp_norm_option]
        intercept = PolyGLMProcessor._pglmprocessor_intercept_options_list[self._pglmprocessor_intercept]

        features = self.covariates

        self._pglmprocessor_pglm = PGLM(features=features, degrees=self._pglmprocessor_degrees, intercept=intercept)
        self._pglmprocessor_deorthonormalization_matrix = treat_data(self._pglmprocessor_pglm)
        return self._pglmprocessor_pglm


    def __user_defined_parameters__(self, fitter):
        return (self._pglmprocessor_intercept, self._pglmprocessor_perp_norm_option) + tuple(
            self._pglmprocessor_degrees)

    def __read_user_defined_parameters__(self, covariate_names, perp_norm_option_global=False,
                                         *args, **kwargs):
        # Intercept term
        # If there are no predictor names, show only options NoIntercept and CorrectionIntercept,
        # and if there are no corrector names, show only NoIntercept and PredictionIntercept. Otherwise,
        # show all options

        default_value = PolyGLMProcessor._pglmprocessor_intercept_options_names[1]
        options_names = PolyGLMProcessor._pglmprocessor_intercept_options_names
        intercept = PolyGLMProcessor._pglmprocessor_intercept_options[super(PolyGLMProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='PolyGLM Processor: How do you want to include the intercept term? (default: {})'.format(
                default_value
            )
        )]

        default_value = PolyGLMProcessor._pglmprocessor_perp_norm_options_names[0]
        options_names = PolyGLMProcessor._pglmprocessor_perp_norm_options_names

        perp_norm_option = \
            PolyGLMProcessor._pglmprocessor_perp_norm_options[super(PolyGLMProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='PolyGLM Processor: How do you want to treat the features? (default: ' +
                      default_value + ')'
        )]

        degrees = []
        for cov in covariate_names:
            degrees.append(super(PolyGLMProcessor, self).__getint__(
                default_value=1,
                lower_limit=1,
                try_ntimes=3,
                show_text='PolyGLM Processor: Please, enter the degree of the feature \'' + str(
                    cov) + '\' (or leave blank to set to 1): '
            ))

        return (intercept, perp_norm_option) + tuple(degrees)

    def __curve__(self, fitter, covariates, covariate_parameters, *args, **kwargs):

        pglm = PGLM(covariates, degrees=self._pglmprocessor_degrees,
                    intercept=PolyGLMProcessor._pglmprocessor_intercept_options_list[self._pglmprocessor_intercept])
        treat_data = GLMProcessor._glmprocessor_perp_norm_options_list[self._pglmprocessor_perp_norm_option]
        treat_data(pglm)

        # Call the normal function with such parameters
        return pglm.predict(covariate_parameters=covariate_parameters)

    def __assign_bound_data__(self, observations, covariates, covariate_parameters, fitting_results):
        # Pre-process parameters for fitter operations (predict, correct, etc.) and leave original
        # parameters for processor operations (curve)
        processed_covariate_parameters = self.__pre_process__(
            covariate_parameters,
            covariates,
        )

        # Assign data to compute AIC
        fitting_results.num_estimated_parameters = self._processor_fitter.num_estimated_parameters(
            covariate_parameters=covariate_parameters,
        )
        fitting_results.max_loglikelihood_value = self._processor_fitter.max_loglikelihood_value(
            observations=fitting_results.corrected_data,
            covariates=covariates,
            covariate_parameters=processed_covariate_parameters,
        )

        bound_functions = ['num_estimated_parameters', 'max_loglikelihood_value']

        # Call parent method
        # bound_functions += super(PolyGLMProcessor, self).__assign_bound_data__(observations, predictors,
        #                                                                        prediction_parameters, correctors,
        #                                                                        correction_parameters, fitting_results)
        return bound_functions

    def get_name(self):
        return 'PolyGLM'


PolyGLMProcessor._pglmprocessor_perp_norm_options = {
    PolyGLMProcessor._pglmprocessor_perp_norm_options_names[i]: i for i in range(
    len(PolyGLMProcessor._pglmprocessor_perp_norm_options_names))
    }
PolyGLMProcessor._pglmprocessor_intercept_options = {
    PolyGLMProcessor._pglmprocessor_intercept_options_names[i]: i for i in range(
    len(PolyGLMProcessor._pglmprocessor_intercept_options_names))
    }
