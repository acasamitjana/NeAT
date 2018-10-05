import numpy as np

from neat.Fitters.GAM import GAM, InterceptSmoother, PolynomialSmoother, SplinesSmoother, SmootherSet, KernelSmoother,\
    TYPE_SMOOTHER, RegressionSplinesSmoother
from neat.Processors.Processing import Processor


class GAMProcessor(Processor):
    _gamprocessor_perp_norm_options_names = [
        'Orthonormalize covariates',
        'Orthogonalize covariates',
        'Normalize covariates',
        'Use correctors and predictors as they are'
    ]

    _gamprocessor_perp_norm_options_list = [
        GAM.orthonormalize_covariates,
        GAM.orthogonalize_covariates,
        GAM.normalize_covariates,
        lambda *args, **kwargs: np.zeros((0, 0))
    ]

    _gamprocessor_intercept_options_names = [
        'Do not include the intercept term',
        'Include the intercept term'
    ]

    _gamprocessor_intercept_options_list = [
        False,
        True
    ]

    _gamprocessor_splines_options_names = RegressionSplinesSmoother._spline_type_list

    _gamprocessor_smoothers_options_names = [smoother.name() for smoother in TYPE_SMOOTHER]




    def __fitter__(self, user_defined_parameters):
        '''Initializes the GAM fitter to be used to process the data.
        '''

        self._gamprocessor_intercept = user_defined_parameters[0]
        self._gamprocessor_perp_norm_option = user_defined_parameters[1]
        self._gamprocessor_smoother_parameters = user_defined_parameters[2]

        sm_index = 0
        covariate_smoothers = SmootherSet()
        for cov in self.covariates.T:
            smoother_function = TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](cov)
            sm_index += 1
            n_param = self._gamprocessor_smoother_parameters[sm_index]
            sm_index += 1
            smoother_function.set_parameters(
                self._gamprocessor_smoother_parameters[sm_index:sm_index + n_param])
            sm_index += n_param
            covariate_smoothers.extend(smoother_function)


        treat_data = GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option]

        gam = GAM(covariate_smoothers=covariate_smoothers, intercept=self._gamprocessor_intercept)

        treat_data(gam)

        return gam

    def __user_defined_parameters__(self, fitter):
        return (self._gamprocessor_intercept, self._gamprocessor_perp_norm_option, self._gamprocessor_smoother_parameters)

    def __read_user_defined_parameters__(self, covariate_names, perp_norm_option_global=False,
                                         *args, **kwargs):

        # Intercept term
        # If there are no predictor names, show only options NoIntercept and CorrectionIntercept,
        # and if there are no corrector names, show only NoIntercept and PredictionIntercept. Otherwise,
        # show all options
        default_value = GAMProcessor._gamprocessor_intercept_options_names[0]
        options_names = GAMProcessor._gamprocessor_intercept_options_names


        intercept = GAMProcessor._gamprocessor_intercept_options[super(GAMProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='GAM Processor: How do you want to include the intercept term? (default: {})'.format(
                default_value
            )
        )]

        if perp_norm_option_global:

            default_value = GAMProcessor._gamprocessor_perp_norm_options_names[0]
            options_names = GAMProcessor._gamprocessor_perp_norm_options_names

            perp_norm_option = GAMProcessor._gamprocessor_perp_norm_options[super(GAMProcessor, self).__getoneof__(
                options_names,
                default_value=default_value,
                show_text='GAM Processor: How do you want to treat the features? (default: ' +
                          default_value + ')'
            )]

        else:
            perp_norm_option = len(GAMProcessor._gamprocessor_perp_norm_options_names) - 1

        print('')
        smoothing_functions = []
        for cov in covariate_names:
            smoother_type = GAMProcessor._gamprocessor_smoothers_options[super(GAMProcessor, self).__getoneof__(
                GAMProcessor._gamprocessor_smoothers_options_names,
                default_value= GAMProcessor._gamprocessor_smoothers_options_names[1],
                show_text='GAM Processor: Please, enter the smoothing function of the feature (covariate) '
                          + str(cov) + ' (default:' + GAMProcessor._gamprocessor_smoothers_options_names[1] + ')'
            )]
            smoothing_functions += [smoother_type]

            if smoother_type == TYPE_SMOOTHER.index(PolynomialSmoother):
                n_params = 1
                polynomial_degree = super(GAMProcessor, self).__getint__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
                              '(or leave blank to set to 3): '
                )

                # Update smoothing functions list
                smoothing_functions += [n_params, polynomial_degree]

            elif smoother_type == TYPE_SMOOTHER.index(SplinesSmoother):
                n_params = 3
                specification_option = super(GAMProcessor, self).__getint__(
                    default_value=0,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Splines smoother. Please, choose if you want to specify degrees of free'
                              'dom (0, by default) or smoothing factor (1): '
                )
                spline_degrees = super(GAMProcessor, self).__getint__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
                              '(or leave blank to set to 3): '
                )
                if specification_option == 0:
                    df_option = super(GAMProcessor, self).__getint__(
                        default_value=spline_degrees + 1,
                        try_ntimes=3,
                        lower_limit=spline_degrees + 1,
                        show_text='GAM Processor: Please, enter the degree of freedom of the spline'
                                  '(or leave it blank to set it to default: spline degree + 1): '
                    )

                    # Update smoothing functions list
                    smoothing_functions += [n_params, specification_option, df_option, spline_degrees]
                else:
                    smoothing_factor = super(GAMProcessor, self).__getfloat__(
                        default_value=500,
                        try_ntimes=3,
                        show_text='GAM Processor: Please, enter the smoothing factor of the spline '
                                  '(or leave it blank to set it to default: 500): '
                    )

                    # Update smoothing functions list
                    smoothing_functions += [n_params, specification_option, smoothing_factor, spline_degrees]

            elif smoother_type == TYPE_SMOOTHER.index(KernelSmoother):
                n_params = 1
                std_kernel = super(GAMProcessor, self).__getfloat__(
                    default_value=1,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Kernel smoother. Please, enter the standard deviation '
                              'of the kernel (or leave blank to set to 1): '
                )

                # Update smoothing functions list
                smoothing_functions += [n_params, std_kernel]

            elif smoother_type == TYPE_SMOOTHER.index(RegressionSplinesSmoother):

                n_params = 2
                splines_type = GAMProcessor._gamprocessor_splines_options[super(GAMProcessor, self).__getoneof__(
                    GAMProcessor._gamprocessor_splines_options_names,
                    default_value = GAMProcessor._gamprocessor_splines_options_names[0],
                    show_text='GAM Processor: You have selected Regression Splines. Please enter the splines type ('
                              'default '+ GAMProcessor._gamprocessor_splines_options_names[0] + ').'
                )]

                print('splines_type' + str(splines_type))
                df = super(GAMProcessor, self).__getfloat__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Regression Splines. Please, enter the desired degrees '
                              'of freedom - minimum df=3. (default: 3): '
                )

                # Update smoothing functions list
                smoothing_functions += [n_params, splines_type, df]


        return (intercept, perp_norm_option, smoothing_functions)

    def __curve__(self, fitter, predictors, prediction_parameters, *args, **kwargs):
        gam = GAM()
        GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option](gam)
        return gam.predict(predictors=predictors, prediction_parameters=prediction_parameters)

    def get_name(self):
        if self._gamprocessor_smoother_parameters[0] == 1:
            smoother_name = 'Poly'
        elif self._gamprocessor_smoother_parameters[0] == 2:
            smoother_name = 'Spline'
        else:
            smoother_name = ''
        return '{}GAM'.format(smoother_name)


GAMProcessor._gamprocessor_perp_norm_options = {GAMProcessor._gamprocessor_perp_norm_options_names[i]: i for i in
                                                range(len(GAMProcessor._gamprocessor_perp_norm_options_names))}

GAMProcessor._gamprocessor_splines_options = {GAMProcessor._gamprocessor_splines_options_names[i]: i for i in
                                                range(len(GAMProcessor._gamprocessor_splines_options_names))}


GAMProcessor._gamprocessor_intercept_options = {GAMProcessor._gamprocessor_intercept_options_names[i]: i for i in
                                                range(len(GAMProcessor._gamprocessor_intercept_options_names))}

GAMProcessor._gamprocessor_smoothers_options = {GAMProcessor._gamprocessor_smoothers_options_names[i]: i for i in
                                                range(len(GAMProcessor._gamprocessor_smoothers_options_names))}

