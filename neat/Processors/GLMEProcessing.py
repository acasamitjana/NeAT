import numpy as np
from neat.Processors.Processing import Processor
from neat.Processors.MixedProcessor import MixedProcessor
from collections import Counter

from neat.Fitters.GLME import GLME
from neat.Fitters.CurveFitting import CombinedFitter


class MixedLongitudinalProcessor(MixedProcessor):

    def __init__(self, subjects, predictors_names, correctors_names, correctors_random_effects_names,
                 predictors, correctors, correctors_random_effects, groups,
                 processing_parameters, user_defined_parameters=(), category=None, type_data='vol',
                 perp_norm_option_global=False):


        """
        Creates a MixedLongitudinalProcessor instance for Longituinal studies. It cast random effects as corretors.

        Parameters
        ----------
        subjects : List<Subject>
            List of subjects to be used in this processor
        predictors_names : List<String>
            List of the names of the features that should be used as predictors
        correctors_names : List<String>
            List of the names of the features that should be used as correctors
        correctors_random_effects_names : List<String>
            List of the names of the features that should be used as correctors
        predictors : numpy.array(NxP)
            Array with the values of the features that should be used as predictors, where N is the number of subjects
            and P the number of predictors
        correctors : numpy.array(NxC)
            Array with the values of the features that should be used as correctors, where N is the number of subjects
            and C the number of correctors
        correctors_random_effects : numpy.array(NxC)
            Array with the values of the features that should be used as correctors, where N is the number of subjects
            and C the number of correctors
        groups : numpy.arrady(NxG)
            Array with the values of each corresponding group of each row (typically, defining several observations
            of the same subject in longitudinal studies). G is the number of groups (typically, number of subjects)
        processing_parameters : dict
            Dictionary with the processing parameters specified in the configuration file, that is, 'mem_usage',
            'n_jobs' and 'cache_size'
        user_defined_parameters : [Optional] tuple
            Parameters passed to the processor. If a empty tuple is passed, the parameters are requested by input.
            Default value: ()
        category : [Optional] String
            Specifies the category for which the fitting should be done. If not specified or None, the fitting is
            computed over all subjects.
            Default value: None
        """

        self._processor_correctors_random_effects_names = correctors_random_effects_names
        self._processor_correctors_random_effects = correctors_random_effects
        self._groups = groups

        super(MixedLongitudinalProcessor, self).__init__(subjects, predictors_names, correctors_names, predictors,
                                                         correctors, processing_parameters,
                                                         user_defined_parameters=user_defined_parameters,
                                                         category=category, type_data=type_data,
                                                         perp_norm_option_global=perp_norm_option_global)

    def __fitter__(self, user_defined_parameters):
        # Store user defined parameters
        self._separate_predictors_by_category = user_defined_parameters[0]
        self._category_predictor_option = user_defined_parameters[1]
        self._perp_norm_option = user_defined_parameters[2]
        self._corrector_udp = user_defined_parameters[3]
        self._predictor_option = user_defined_parameters[4]
        self._predictor_udp = user_defined_parameters[5]

        # Separate predictors as required
        all_categories = [subj.category for subj in self._processor_subjects]
        c = Counter(all_categories)
        if self._separate_predictors_by_category:
            # Create NxM ndarray, with the same number of samples as the original predictors array but with
            # M columns, where M is the number of categories, and put 0 where corresponding
            M = len(c)
            N = len(self._processor_subjects)
            predictors_array = np.zeros((N, M))
            for index, category in enumerate(list(c)):
                category_index = [i for i, x in enumerate(all_categories) if x == category]
                selected_predictors = self._processor_predictors[category_index, 0]
                predictors_array[category_index, index] = selected_predictors

            # Separate the category to be treated as a predictor from the others,
            # which will be trated as correctors
            cat_predictor = [pos for pos, x in enumerate(list(c)) if self._category_predictor_option == x][0]
            cat_corrector = [pos for pos, x in enumerate(list(c)) if self._category_predictor_option != x]
            self._processor_predictors = np.atleast_2d(predictors_array[:, cat_predictor]).T

            self._processor_correctors = np.concatenate(
                (self._processor_correctors, predictors_array[:, cat_corrector]),
                axis=1
            )
            # Change predictors and correctors names
            original_predictor_name = self._processor_predictors_names[0]
            self._processor_predictors_names = [
                original_predictor_name + ' (category {})'.format(self._category_predictor_option)
            ]
            self._processor_correctors_names += [
                original_predictor_name + ' (category {})'.format(cat) for cat in cat_corrector
                ]

        # Create correction processor
        self._correction_processor = GLMEProcessor(self._processor_subjects, [], self._processor_correctors_names,
                                                   self._processor_correctors_random_effects_names,
                                                   np.zeros((len(self._processor_subjects), 0)),
                                                   self._processor_correctors,
                                                   self._processor_correctors_random_effects, self._groups,
                                                   self._processor_processing_params, tuple(self._corrector_udp),
                                                   type_data=self._type_data)

        # Create prediction processor
        # self._prediction_processor = MixedProcessor._mixedprocessor_processor_list[
        #     self._predictor_option
        # ](self._processor_subjects, self._processor_predictors_names, [], self._processor_predictors, np.zeros((len(self._processor_subjects), 0)),
        #   self._processor_processing_params, tuple(self._predictor_udp),type_data=self._type_data)

        # Get correction fitter
        correction_fitter = self._correction_processor.fitter
        prediction_fitter = self._prediction_processor.fitter

        # Create MixedFitter
        fitter = CombinedFitter(correction_fitter, prediction_fitter)()
        treat_data = MixedProcessor._mixedprocessor_perp_norm_options_list[self._perp_norm_option]
        self._mixedprocessor_deorthonormalization_matrix = treat_data(fitter)

        return fitter

    def __user_defined_parameters__(self, fitter):
        return self._separate_predictors_by_category, \
               self._category_predictor_option, \
               self._perp_norm_option, \
               self._corrector_udp, \
               self._predictor_option, \
               self._predictor_udp

    def __read_user_defined_parameters__(self, predictor_names, corrector_names, *args, **kwargs):

        # Mixed processor options
        separate_predictors_by_category = False
        category_predictor_option = 'All'
        all_categories = [subj.category for subj in self._processor_subjects]
        c = Counter(all_categories)
        if (self._category is None) and (None not in all_categories):
            # Ask user to separate predictors if there is no category specified for this processor
            separate_predictors_by_category = MixedProcessor.__getyesorno__(
                default_value=False,
                try_ntimes=3,
                show_text='\nMixedProcessor: Do you want to separate the predictor by categories? (Y/N, default N): '
            )
            if separate_predictors_by_category:
                # Ask which category should remain in predictors
                options_list = list(c)
                category_predictor_option = MixedProcessor.__getoneof__(
                    option_list=options_list,
                    default_value=0,
                    try_ntimes=3,
                    show_text='MixedProcessor: Which category do you want to have as a predictor, thus being the rest '
                              'correctors? (default value: 0)'
                )

                cat_predictor = [pos for pos, x in enumerate(list(c)) if category_predictor_option == x]
                cat_corrector = [pos for pos, x in enumerate(list(c)) if category_predictor_option != x]
                # Change predictors and correctors names
                original_predictor_name = predictor_names[0]
                predictor_names = [
                    original_predictor_name + ' (category {})'.format(cat) for cat in cat_predictor
                    ]
                corrector_names += [
                    original_predictor_name + ' (category {})'.format(cat) for cat in cat_corrector
                    ]


        # Correction fitter --> GLME
        print('MixedProcessor: In longitudinal studies, only a Longitudinal Linear Mixed Effects modeling is available')

        # Prediction fitter
        keys = list(MixedProcessor._mixedprocessor_processor_options.keys())
        keys.sort()
        predict_option_name = MixedProcessor.__getoneof__(
            keys,
            default_value='Poly GLM',
            try_ntimes=3,
            show_text='MixedProcessor: Select the fitter to be used for prediction '
                      '(default value: Poly GLM)'
        )
        predict_option = MixedProcessor._mixedprocessor_processor_options[predict_option_name]

        print()
        print("--------------------------------")
        print(" TREAT DATA if both are PolyGLM ")
        print("--------------------------------")
        if predict_option == MixedProcessor._mixedprocessor_processor_options['Poly GLM']:
            perp_norm_option_global = MixedProcessor._mixedprocessor_perp_norm_options[
                super(MixedProcessor, self).__getoneof__(
                    MixedProcessor._mixedprocessor_perp_norm_options_names[:4],
                    default_value=MixedProcessor._mixedprocessor_perp_norm_options_names[0],
                    show_text='PolyGLM Processor: How do you want to treat the features? (default: ' +
                              MixedProcessor._mixedprocessor_perp_norm_options_names[0] + ')'
                )]

        else:
            perp_norm_option_global = 3

        print()
        print("---------------------")
        print(" CORRECTOR PARAMETERS")
        print("---------------------")
        # Create dummy array with proper dimensions to pass it as correctors to be the same size as the names
        N = len(self._processor_subjects)
        M = len(corrector_names)
        correctors = np.zeros((N, M))
        correct_processor = GLMEProcessor(self._processor_subjects, [], corrector_names,
                                           self._processor_correctors_random_effects_names, np.zeros((0, 0)),
                                           correctors, self._processor_correctors_random_effects, self._groups,
                                           self._processor_processing_params,
                                           perp_norm_option_global=(perp_norm_option_global == 3))

        correct_udp = list(correct_processor.user_defined_parameters)

        print()
        print("----------------------")
        print(" PREDICTOR PARAMETERS")
        print("----------------------")
        # Create dummy array with proper dimensions to pass it as correctors to be the same size as the names
        M = len(predictor_names)
        predictors = np.zeros((N, M))
        # User defined parameters for correction fitter
        predict_processor = MixedProcessor._mixedprocessor_processor_list[
            predict_option
        ](self._processor_subjects, predictor_names, [], predictors, np.zeros((0, 0)),
          self._processor_processing_params, perp_norm_option_global=(perp_norm_option_global == 3))
        predict_udp = list(predict_processor.user_defined_parameters)

        return separate_predictors_by_category, category_predictor_option, perp_norm_option_global, \
               correct_udp, predict_option, predict_udp


class GLMEProcessor(Processor):

    _glmeprocessor_perp_norm_options_names = [
        'Orthonormalize all',
        'Orthogonalize all',
        'Normalize all',

        'Orthonormalize predictors',
        'Orthogonalize predictors',
        'Normalize predictors',

        'Orthonormalize correctors',
        'Orthogonalize correctors',
        'Normalize correctors',
        'Use predictors and/or correctors as they are'
    ]

    _glmeprocessor_perp_norm_options_list = [
        GLME.orthonormalize_all,
        GLME.orthogonalize_all,
        GLME.normalize_all,
        GLME.orthonormalize_predictors,
        GLME.orthogonalize_predictors,
        GLME.normalize_predictors,
        GLME.orthonormalize_correctors,
        GLME.orthogonalize_correctors,
        GLME.normalize_correctors,
        lambda *args, **kwargs: np.zeros((0, 0))
    ]

    _glmeprocessor_intercept_options_names = [
        'Do not include the intercept term',
        'As a corrector',
        'As a predictor'
    ]

    _glmeprocessor_intercept_options_list = [
        GLME.NoIntercept,
        GLME.CorrectionIntercept,
        GLME.PredictionIntercept
    ]

    _glmeprocessor_submodels_options_names = [
        'Do not include this term in the system',
        'As a corrector',
        'As a predictor'
    ]


    def __init__(self, subjects, predictors_names, correctors_names, correctors_random_effects_names,
                 predictors, correctors, correctors_random_effects, groups,
                 processing_parameters, user_defined_parameters=(), category=None, type_data='vol',
                 perp_norm_option_global=False):
        """
        Creates a Processor instance

        Parameters
        ----------
        subjects : List<Subject>
            List of subjects to be used in this processor
        predictors_names : List<String>
            List of the names of the features that should be used as predictors
        correctors_names : List<String>
            List of the names of the features that should be used as correctors
        predictors : numpy.array(NxP)
            Array with the values of the features that should be used as predictors, where N is the number of subjects
            and P the number of predictors
        correctors : numpy.array(NxC)
            Array with the values of the features that should be used as correctors, where N is the number of subjects
            and C the number of correctors
        processing_parameters : dict
            Dictionary with the processing parameters specified in the configuration file, that is, 'mem_usage',
            'n_jobs' and 'cache_size'
        user_defined_parameters : [Optional] tuple
            Parameters passed to the processor. If a empty tuple is passed, the parameters are requested by input.
            Default value: ()
        category : [Optional] String
            Specifies the category for which the fitting should be done. If not specified or None, the fitting is
            computed over all subjects.
            Default value: None
        """


        self._processor_correctors_random_effects_names = correctors_random_effects_names
        self._processor_correctors_random_effects = correctors_random_effects
        self._groups = groups

        super(GLMEProcessor,self).__init__(subjects, predictors_names, correctors_names, predictors,
                                                     correctors, processing_parameters,
                                                     user_defined_parameters=user_defined_parameters,
                                                     category=category,  type_data=type_data,
                                                     perp_norm_option_global=perp_norm_option_global)

    def __fitter__(self, user_defined_parameters):
        self._glmeprocessor_intercept_fe = user_defined_parameters[0]
        self._glmeprocessor_intercept_re = user_defined_parameters[1]
        self._glmeprocessor_perp_norm_option = user_defined_parameters[2]
        self._glmeprocessor_degrees = user_defined_parameters[3:]

        # Orthonormalize/Orthogonalize/Do nothing options
        treat_data = GLMEProcessor._glmeprocessor_perp_norm_options_list[self._glmeprocessor_perp_norm_option]
        # Intercept option
        intercept_fe = GLMEProcessor._glmeprocessor_intercept_options_list[self._glmeprocessor_intercept_fe]
        intercept_re = GLMEProcessor._glmeprocessor_intercept_options_list[self._glmeprocessor_intercept_re]

        # Construct data matrix from correctors and predictor
        num_regs = self.predictors.shape[1]
        num_correc = self.correctors.shape[1]
        features = np.zeros((self.predictors.shape[0], num_regs + num_correc))
        features[:, :num_regs] = self.predictors
        features[:, num_regs:] = self.correctors

        # Instantiate a PolySVR
        glme = GLME()
        treat_data(glme)
        return glme

    def __read_user_defined_parameters__(self, predictor_names, corrector_names, perp_norm_option_global=False,
                                         *args, **kwargs):
    
        # Intercept term
        # If there are no predictor names, show only options NoIntercept and CorrectionIntercept,
        # and if there are no corrector names, show only NoIntercept and PredictionIntercept. Otherwise,
        # show all options
        if len(predictor_names) == 0:
            default_value = GLMEProcessor._glmeprocessor_intercept_options_names[1]
            options_names = GLMEProcessor._glmeprocessor_intercept_options_names[:2]
        elif len(corrector_names) == 0:
            default_value = GLMEProcessor._glmeprocessor_intercept_options_names[2]
            options_names = GLMEProcessor._glmeprocessor_intercept_options_names[::2]
        else:
            default_value = GLMEProcessor._glmeprocessor_intercept_options_names[1]
            options_names = GLMEProcessor._glmeprocessor_intercept_options_names

        intercept_fe = GLMEProcessor._glmeprocessor_intercept_options[super(GLMEProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='GLME Processor: How do you want to include the fixed-effect intercept term? (default: {})'.format(
                default_value
            )
        )]

        intercept_re = GLMEProcessor._glmeprocessor_intercept_options[super(GLMEProcessor, self).__getyesorno__(
            options_names,
            default_value='Y',
            show_text='GLME Processor: Do you want to include the random-effects intercept? (Y/N, default Y)'.format(
                'Y'
            )
        )]


        
        if perp_norm_option_global:
            if len(predictor_names) == 0:
                default_value = GLMEProcessor._glmeprocessor_perp_norm_options_names[6]
                options_names = GLMEProcessor._glmeprocessor_perp_norm_options_names[6:]
            elif len(corrector_names) == 0:
                default_value = GLMEProcessor._glmeprocessor_perp_norm_options_names[3]
                options_names = GLMEProcessor._glmeprocessor_perp_norm_options_names[3:6] + \
                                GLMEProcessor._glmeprocessor_perp_norm_options_names[-1:]
            else:
                default_value = GLMEProcessor._glmeprocessor_perp_norm_options_names[0]
                options_names = GLMEProcessor._glmeprocessor_perp_norm_options_names
        
            perp_norm_option = GLMEProcessor._glmeprocessor_perp_norm_options[super(GLMEProcessor, self).__getoneof__(
                options_names,
                default_value=default_value,
                show_text='GLME Processor: How do you want to treat the fixed-effects features? (default: ' +
                          default_value + ')'
            )]
        
        else:
            perp_norm_option = 8
        
        degrees = []
        for reg in predictor_names:
            degrees.append(super(GLMEProcessor, self).__getint__(
                default_value=1,
                lower_limit=1,
                try_ntimes=3,
                show_text='GLME Processor: Please, enter the degree of the feature (predictor) \'' + str(
                    reg) + '\' (or leave blank to set to 1): '
            ))
        for cor in corrector_names:
            degrees.append(super(GLMEProcessor, self).__getint__(
                default_value=1,
                try_ntimes=3,
                show_text='GLME Processor: Please, enter the degree of the feature (corrector) \'' + str(
                    cor) + '\' (or leave blank to set to 1): '
            ))

        
        return (intercept_fe, intercept_re, perp_norm_option) + tuple(degrees)



GLMEProcessor._glmeprocessor_perp_norm_options = {
    GLMEProcessor._glmeprocessor_perp_norm_options_names[i]: i for i in range(
    len(GLMEProcessor._glmeprocessor_perp_norm_options_names))
    }
GLMEProcessor._glmeprocessor_intercept_options = {
    GLMEProcessor._glmeprocessor_intercept_options_names[i]: i for i in range(
    len(GLMEProcessor._glmeprocessor_intercept_options_names))
    }
