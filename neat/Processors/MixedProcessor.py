from collections import Counter

import numpy as np
import os

from neat.Fitters.CurveFitting import CombinedFitter, CurveFitter
from neat.Fitters.GLM import GLM
from neat.Processors.GAMProcessing import GAMProcessor
from neat.Processors.GLMProcessing import PolyGLMProcessor
from neat.Processors.Processing import Processor, NullProcessor
from neat.Processors.SVRProcessing import PolySVRProcessor, GaussianSVRProcessor
from neat.Utils.Transforms import orthogonalize_all, orthonormalize_all, normalize_all

from neat.Utils.Subject import Chunks
from neat.FitScores.FitEvaluation import evaluation_function as eval_func, FittingResults
from neat.FitScores.FitEvaluation import effect_strength as eff_size_eval, effect_type as eff_type_eval
from neat.Utils.Math import find_non_orthogonal_columns

class MixedProcessor(Processor):
    """
    Processor that uses MixedFitter to allow you to correct and predict the data with two
    different fitters
    """
    __threshold = 1e-14

    # Available processors
    _mixedprocessor_processor_list = [
        PolyGLMProcessor,
        GAMProcessor,
        PolySVRProcessor,
        GaussianSVRProcessor,
    ]

    _mixedprocessor_processor_options = {
        'Poly GLM': 0,
        'GAM': 1,
        'Poly SVR': 2,
        'Gaussian SVR': 3,
    }

    _mixedprocessor_perp_norm_options_names = [
        'Orthonormalize all',
        'Orthogonalize all',
        'Normalize all',
        'Treat restricted and predictor models independently',

    ]

    _mixedprocessor_perp_norm_options_list = [
        Processor.orthonormalize_all,
        Processor.orthogonalize_all,
        Processor.normalize_all,
        lambda x: np.zeros((0,0)),
    ]

    class Results:
        def __init__(self, prediction_parameters, correction_parameters):  # , fitting_scores):
            self._prediction_parameters = prediction_parameters
            self._correction_parameters = correction_parameters

        @property
        def prediction_parameters(self):
            return self._prediction_parameters

        @property
        def correction_parameters(self):
            return self._correction_parameters

        def __str__(self):
            s = 'Results:'
            s += '\n    Correction parameters:' + reduce(lambda x, y: x + '\n    ' + y,
                                                         repr(self._correction_parameters).split('\n'))
            s += '\n\n    Prediction parameters:' + reduce(lambda x, y: x + '\n    ' + y,
                                                           repr(self._prediction_parameters).split('\n'))
            return s

    def __fitter__(self, user_defined_parameters):
        # Store user defined parameters
        self._separate_covariates_by_category = user_defined_parameters[0]
        self._separate_covariates_by_category_list = user_defined_parameters[1]
        self._perp_norm_option = user_defined_parameters[2]
        self._contrast = user_defined_parameters[3]
        self._corrector_option = user_defined_parameters[4]
        self._corrector_udp = user_defined_parameters[5]
        self._predictor_option = user_defined_parameters[6]
        self._predictor_udp = user_defined_parameters[7]

        # Separate covariates as required
        all_categories = [subj.category for subj in self._processor_subjects]
        c = Counter(all_categories)
        if self._separate_covariates_by_category:
            # Create NxM ndarray, with the same number of samples as the original predictors array but with
            # M columns, where M is the number of categories, and put 0 where corresponding
            M = self._processor_covariates.shape[1] + len(self._separate_covariates_by_category_list)*(len(c)-1)
            N = len(self._processor_subjects)
            covariates_array = np.zeros((N, M))
            it_m = 0
            for it_covariate in range(self._processor_covariates.shape[1]):
                if it_covariate in self._separate_covariates_by_category_list:
                    for index, category in enumerate(list(c)):
                        category_index = [i for i, x in enumerate(all_categories) if x == category]
                        selected_predictors = self._processor_covariates[category_index, it_covariate]
                        covariates_array[category_index, it_m] = selected_predictors
                        print(selected_predictors)
                        it_m+=1
                else:
                    covariates_array[:, it_m] = self._processor_covariates[:, it_covariate]
                    it_m += 1

            category_list = list(c)

            original_covariates_names = self._processor_covariates_names
            covariates_names = []
            for it_cov_name, cov_name in enumerate(original_covariates_names):
                if cov_name in self._separate_covariates_by_category_list:
                    covariates_names += [
                        cov_name + ' (category {})'.format(cat) for cat in category_list
                        ]
                else:
                    covariates_names += [cov_name]

            self._processor_covariates_names = covariates_names
            self._processor_covariates = covariates_array

        else:
            covariates_names = self._processor_covariates_names


        #Operations with contrasts
        contrast = np.asarray(self._contrast).T

        number_of_contrasts = contrast.shape[1]
        contrast_null_full = np.eye(len(covariates_names)) - np.dot(contrast, np.linalg.pinv(contrast)) #There might be linearly dependent rows
        lambda_null = find_non_orthogonal_columns(contrast_null_full)

        contrast_null = np.zeros((len(covariates_names), np.linalg.matrix_rank(contrast_null_full)))
        it_cn = 0
        for it_l in range(contrast_null_full.shape[1]):
            if it_l not in lambda_null:
                contrast_null[:,it_cn] = contrast_null_full[:,it_l]
                it_cn +=1


        #Covariate

        treat_data = MixedProcessor._mixedprocessor_perp_norm_options_list[self._perp_norm_option]
        treat_data(self)
        predictors = np.dot(self.covariates,contrast)
        correctors = np.dot(self.covariates,contrast_null)
        self._processor_covariates = np.concatenate((correctors, predictors),axis=1)
        self._processor_correctors, self._processor_predictors = self.covariates[:,:-number_of_contrasts],self.covariates[:,-number_of_contrasts:]


        # Covariate names
        n_predictors = np.linalg.matrix_rank(contrast)
        n_correctors = np.linalg.matrix_rank(contrast_null)

        predictor_names = []
        corrector_names = []
        if n_predictors == number_of_contrasts:
            for nc in range(number_of_contrasts):
                predictor_name_tmp = [str(c_i) + '-'+ covariates_names[it_c_i] for it_c_i, c_i in enumerate(contrast[:, nc]) if c_i != 0]
                predictor_names.append('_'.join(predictor_name_tmp))
        else:
            raise ValueError("[MixedProcessor]: Please, specify orthogonal contrasts")

        for it_l in range(n_correctors):
            corrector_name_tmp = [ str(c_i)+'-'+covariates_names[it_c_i] for it_c_i, c_i in enumerate(contrast_null[:,it_l]) if c_i != 0]
            corrector_names.append('_'.join(corrector_name_tmp))


        if len(covariates_names) != n_predictors + n_correctors:
            raise ValueError(
                "[MixedProcessor]: There has been an error when computing the contrasts of the model")

        self._processor_predictors_names = predictor_names
        self._processor_correctors_names = corrector_names

        ############################ Using single contrast (a delta vector in the variable of interest)
        # self._processor_predictors_names = [c_name for it_c_name, c_name in enumerate(covariates_names) if it_c_name in self._specify_predictor_variables]
        # self._processor_correctors_names = [c_name for it_c_name, c_name in enumerate(covariates_names) if it_c_name not in self._specify_predictor_variables]
        #
        # self._processor_correctors = np.zeros((covariates_names.shape[0], len(self._specify_predictor_variables)))
        # self._processor_predictors = np.zeros((covariates_names.shape[0],
        #                                        covariates_names.shape[1] - len(self._specify_predictor_variables)))
        ############################


        # Create correction processor
        self._correction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._corrector_option
        ](self._processor_subjects, self._processor_correctors_names, self._processor_correctors,
          self._processor_processing_params, tuple(self._corrector_udp),type_data=self._type_data)


        # Create prediction processor
        self._prediction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._predictor_option
        ](self._processor_subjects, self._processor_predictors_names, self._processor_predictors,
          self._processor_processing_params, tuple(self._predictor_udp),type_data=self._type_data)

        # Get correction fitter
        correction_fitter = self._correction_processor.fitter
        prediction_fitter = self._prediction_processor.fitter

        # Create MixedFitter
        fitter = CombinedFitter(correction_fitter, prediction_fitter)

        return fitter

    def __user_defined_parameters__(self, fitter):
        return self._separate_covariates_by_category, \
               self._separate_covariates_by_category_list, \
               self._perp_norm_option, \
               self._contrast, \
               self._corrector_option, \
               self._corrector_udp, \
               self._predictor_option, \
               self._predictor_udp

    def __read_user_defined_parameters__(self, covariates_names,  *args, **kwargs):

        # Mixed processor options
        separate_covariates_by_category = False
        separate_covariates_by_category_list = []
        all_categories = [subj.category for subj in self._processor_subjects]
        c = Counter(all_categories)
        if (self._category is None) and (None not in all_categories):
            # Ask user to separate predictors if there is no category specified for this processor

            separate_covariates_by_category = MixedProcessor.__getyesorno__(
                default_value=False,
                try_ntimes=3,
                show_text='\nMixedProcessor: Do you want to separate covariates by categories? (Y/N, default N): '
            )


            if separate_covariates_by_category:
                separate_covariates_by_category_list = MixedProcessor.__getmultipleof__(
                    option_list=covariates_names,
                    try_ntimes=3,
                    default_value=covariates_names,
                    show_text='\nMixedProcessor: Which covariates do you want to separate by categories? (List with separating commas, default ALL): '
                )

                category_list = list(c)
                # Change predictors and correctors names
                original_covariates_names = covariates_names
                covariates_names = []
                for it_cov_name, cov_name in enumerate(original_covariates_names):
                    if cov_name in separate_covariates_by_category_list:
                        covariates_names += [
                            cov_name + ' (category {})'.format(cat) for cat in category_list
                        ]
                    else:
                        covariates_names += [cov_name]


        print()
        print("--------------------------------")
        print(" TREAT DATA")
        print("--------------------------------")

        perp_norm_option_global = MixedProcessor._mixedprocessor_perp_norm_options[
            super(MixedProcessor, self).__getoneof__(
                MixedProcessor._mixedprocessor_perp_norm_options_names,
                default_value=MixedProcessor._mixedprocessor_perp_norm_options_names[0],
                show_text='PolyGLM Processor: How do you want to treat the features? (default: ' +
                          MixedProcessor._mixedprocessor_perp_norm_options_names[0] + ')'
            )]


        number_of_contrasts = MixedProcessor.__getint__(
            default_value=1,
            try_ntimes=3,
            lower_limit=0,
            show_text='MixedProcessor: How many contrasts do you want to apply? (default value: 1)'
        )

        if number_of_contrasts == 0:
            contrast = 0
        else:
            print('\n')
            print("Let's specify different contrasts with these covariates:")
            for cn in covariates_names:
                print(' - ' + cn)
            print('\n')

            contrast = np.zeros((len(covariates_names),number_of_contrasts))
            for nc in range(number_of_contrasts):
                contrast[:,nc] = MixedProcessor.__getlist__(length=len(covariates_names),
                                                            try_ntimes=3,
                                                            show_text='MixedProcessor: Specify contrast ' + str(nc),
                                                            obtain_input_from=input,
                                                            )

        contrast_null = np.eye(len(covariates_names)) - np.dot(contrast,np.linalg.pinv(contrast))

        n_predictors = np.linalg.matrix_rank(contrast)
        n_correctors = np.linalg.matrix_rank(contrast_null)

        predictor_names = []
        corrector_names = []

        if n_predictors == number_of_contrasts:
            for nc in range(number_of_contrasts):
                predictor_name_tmp = [ str(c_i)+'-'+covariates_names[it_c_i] for it_c_i, c_i in enumerate(contrast[:,nc]) if c_i != 0]
                predictor_names.append('_'.join(predictor_name_tmp))
        else:
            raise ValueError("[MixedProcessor]: Please, specify orthogonal contrasts")

        lambda_null = find_non_orthogonal_columns(contrast_null)
        for it_l in range(contrast_null.shape[1]):
            if it_l not in lambda_null:
                corrector_name_tmp = [ str(c_i)+'-'+covariates_names[it_c_i] for it_c_i, c_i in enumerate(contrast_null[:,it_l]) if c_i != 0]
                corrector_names.append('_'.join(corrector_name_tmp))

        if len(corrector_names) != n_correctors:
            raise ValueError("[MixedProcessor]: There has been an error when computing the null contrast for the restricted model")

        # predictor_name = [c for it_c, c in enumerate(contrast) if it_c in specify_predictor_variables]
        # corrector_names = [c_name for it_c_name, c_name in enumerate(covariates_names) if it_c_name not in specify_predictor_variables]

        # Correction fitter
        keys = list(MixedProcessor._mixedprocessor_processor_options.keys())
        keys.sort()
        correct_option_name = MixedProcessor.__getoneof__(
            keys,
            default_value='Poly GLM',
            try_ntimes=3,
            show_text='MixedProcessor: Select the fitter to be used for correction '
                      '(default value: Poly GLM)'
        )
        correct_option = MixedProcessor._mixedprocessor_processor_options[correct_option_name]

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

        # print()
        # print("--------------------------------")
        # print(" TREAT DATA if both are PolyGLM ")
        # print("--------------------------------")
        # if correct_option == predict_option and predict_option == MixedProcessor._mixedprocessor_processor_options['Poly GLM']:
        #     perp_norm_option_global = MixedProcessor._mixedprocessor_perp_norm_options[
        #         super(MixedProcessor, self).__getoneof__(
        #             MixedProcessor._mixedprocessor_perp_norm_options_names,
        #             default_value=MixedProcessor._mixedprocessor_perp_norm_options_names[0],
        #             show_text='PolyGLM Processor: How do you want to treat the features? (default: ' +
        #                       MixedProcessor._mixedprocessor_perp_norm_options_names[0] + ')'
        #         )]
        #
        # else:
        #     perp_norm_option_global = 3

        print()
        print( "---------------------")
        print( " CORRECTOR PARAMETERS")
        print( "---------------------")
        # Create dummy array with proper dimensions to pass it as correctors to be the same size as the names
        N = len(self._processor_subjects)
        M = len(corrector_names)
        correctors = np.zeros((N, M))
        # User defined parameters for correction fitter
        # if correct_option == -1:
        #     correct_processor = NullProcessor(self._processor_subjects, [], corrector_names, np.zeros((0, 0)), correctors,
        #   self._processor_processing_params, perp_norm_option_global=(perp_norm_option_global==3))
        # else:
        correct_processor = MixedProcessor._mixedprocessor_processor_list[
        correct_option
        ](self._processor_subjects, corrector_names, correctors,self._processor_processing_params)

        correct_udp = list(correct_processor.user_defined_parameters)

        print()
        print( "----------------------")
        print( " PREDICTOR PARAMETERS")
        print( "----------------------")
        # Create dummy array with proper dimensions to pass it as correctors to be the same size as the names
        M = len(predictor_names)
        predictors = np.zeros((N, M))
        # User defined parameters for correction fitter
        predict_processor = MixedProcessor._mixedprocessor_processor_list[
            predict_option
        ](self._processor_subjects, predictor_names, predictors, self._processor_processing_params)
        predict_udp = list(predict_processor.user_defined_parameters)


        return separate_covariates_by_category, separate_covariates_by_category_list, perp_norm_option_global, \
               contrast.T.tolist(), correct_option, correct_udp, predict_option, predict_udp

    def __post_process__(self, prediction_parameters, correction_parameters):
        # Route post-processing routines to the corresponding processors
        prediction_results = self._prediction_processor.__post_process__(
            prediction_parameters
        )
        correction_results = self._correction_processor.__post_process__(
            correction_parameters
        )


        # Return the post_processed parameters
        return MixedProcessor.Results(
            prediction_results.covariates_parameters,
            correction_results.covariates_parameters
        )

    def __pre_process__(self, prediction_parameters, correction_parameters, predictors, correctors):
        # Route pre-processing routines to the corresponding processors
        pparams = self._prediction_processor.__pre_process__(
            prediction_parameters,
            predictors,
        )

        cparams = self._correction_processor.__pre_process__(
            correction_parameters,
            correctors
        )

        return pparams, cparams

    def __curve__(self, fitter, predictor, prediction_parameters, *args, **kwargs):
        return self._prediction_processor.__curve__(fitter, predictor, prediction_parameters)

    def __corrected_values__(self, observations, correction_parameters, *args, **kwargs):
        return self.fitter.correct(observations = observations,
                                   correction_parameters = correction_parameters,
                                   *args, **kwargs)

    def process(self, x1=0, x2=None, *args, **kwargs):
        """
        Processes all the data from coordinates x1, y1, z1 to x2, y2, z2

        Parameters
        ----------
        x1 : int
            Voxel in the x-axis from where the processing begins
        x2 : int
            Voxel in the x-axis where the processing ends
        y1 : int
            Voxel in the y-axis from where the processing begins
        y2 : int
            Voxel in the y-axis where the processing ends
        z1 : int
            Voxel in the z-axis from where the processing begins
        z2 : int
            Voxel in the z-axis where the processing ends
        args : List
        kwargs : Dict

        Returns
        -------
        Processor.Results instance
            Object with two properties: correction_parameters and prediction_parameters
        """
        chunks = Chunks(
            self._processor_subjects, x1=x1, x2=x2,
            mem_usage=self._processor_processing_params['mem_usage']
        )
        dims = chunks.dims

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Add processing parameters to kwargs
        kwargs['n_jobs'] = self._processor_processing_params['n_jobs']
        kwargs['cache_size'] = self._processor_processing_params['cache_size']

        # Get the results of the first chunk to initialize dimensions of the solution matrices
        # Get first chunk and fit the parameters
        chunk = next(chunks)

        self._processor_fitter.fit(chunk.data, *args, **kwargs)

        # Get the parameters and the dimensions of the solution matrices
        cparams = self._processor_fitter.correction_parameters
        pparams = self._processor_fitter.prediction_parameters
        cpdims = tuple(cparams.shape[:-1] + dims)
        rpdims = tuple(pparams.shape[:-1] + dims)

        # Initialize solution matrices
        correction_parameters = np.zeros(cpdims, dtype=np.float64)
        prediction_parameters = np.zeros(rpdims, dtype=np.float64)

        # Assign first chunk's solutions to solution matrices
        dx = cparams.shape[-1]
        correction_parameters[:, :dx] = cparams
        prediction_parameters[:, :dx] = pparams

        # Update progress
        self._processor_update_progress(prog_inc * dx)

        # Now do the same for the rest of the Chunks
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x = chunk.coords
            x -= x1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx = cdata.shape[-1]

            # Fit the parameters to the data in the chunk
            self._processor_fitter.fit(cdata, *args, **kwargs)

            # Get the optimal parameters and insert them in the solution matrices
            correction_parameters[:, x:x + dx] = self._processor_fitter.correction_parameters
            prediction_parameters[:, x:x + dx] = self._processor_fitter.prediction_parameters

            # Update progress
            self._processor_update_progress(prog_inc * dx)

        if self.progress != 100.0:
            self._processor_update_progress(10000.0 - self._processor_progress)

        # Call post_processing routine
        return self.__post_process__(prediction_parameters, correction_parameters)

    def assign_bound_data(self, observations, predictors, prediction_parameters, correctors,
                          correction_parameters,fitting_results):

        # Restrictive bound data assignment: only if both processors are instances of the same class call their
        # specific implementation of __assign_bound_data__

        processed_prediction_parameters, processed_correction_parameters = self.__pre_process__(
            prediction_parameters,
            correction_parameters,
            predictors,
            correctors
        )

        fitting_results.observations = observations
        fitting_results.corrected_data = self._processor_fitter.correct(
            observations=observations,
            correctors=correctors,
            correction_parameters=processed_correction_parameters
        )
        fitting_results.predicted_data = self._processor_fitter.predict(
            predictors=predictors,
            prediction_parameters=processed_prediction_parameters
        )
        fitting_results.df_correction = self._processor_fitter.df_correction(
            observations=observations,
            correctors=correctors,
            correction_parameters=processed_correction_parameters
        )
        fitting_results.df_prediction = self._processor_fitter.df_prediction(
            observations=observations,
            predictors=predictors,
            prediction_parameters=processed_prediction_parameters
        )
        axis, curve = self.curve(
            covariate_parameters=prediction_parameters,
            tpoints=2 * len(self.subjects)
        )

        fitting_results.curve = curve
        fitting_results.xdiff = axis[..., 1] - axis[..., 0]

        bound_functions = ['observations', 'corrected_data', 'predicted_data', 'df_correction', 'df_prediction', 'curve', 'xdiff']


        bound_functions += self._prediction_processor.__assign_bound_data__(observations,
                                                                            predictors,
                                                                            prediction_parameters,
                                                                            fitting_results
                                                                            )


        return bound_functions

    def get_name(self):
        corrector_name = 'correction_{}'.format(self._correction_processor.get_name())
        predictor_name = 'prediction_{}'.format(self._prediction_processor.get_name())
        processor_name = '{}-{}'.format(corrector_name, predictor_name)
        if self._category is not None:
            processor_name += '-category_{}'.format(self._category)
        return processor_name


    @property
    def corrector_names(self):
        """
        Matrix of correctors of this instance

        Returns
        -------
        numpy.array (NxC)
            Values of the features of the subjects that are to be used as correctors in the fitter, where N is the
            number of subjects and C the number of correctors
        """
        return self._processor_correctors_names

    @property
    def predictor_names(self):
        """
        Matrix of predictors of this instance.

        Returns
        -------
        numpy.array (NxR)
            Values of the features of the subjects that are to be used as predictors in the fitter, where N is the
            number of subjects and R the number of predictors
        """
        return self._processor_predictors_names

    @property
    def correctors(self):
        """
        Matrix of correctors of this instance

        Returns
        -------
        numpy.array (NxC)
            Values of the features of the subjects that are to be used as correctors in the fitter, where N is the
            number of subjects and C the number of correctors
        """
        return self._processor_correctors

    @property
    def predictors(self):
        """
        Matrix of predictors of this instance.

        Returns
        -------
        numpy.array (NxR)
            Values of the features of the subjects that are to be used as predictors in the fitter, where N is the
            number of subjects and R the number of predictors
        """
        return self._processor_predictors

    @property
    def correction_processor(self):
        return self._correction_processor

    @property
    def prediction_processor(self):
        return self._prediction_processor


class MixedVolumeProcessor(MixedProcessor):

    def process(self, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, *args, **kwargs):
        """
        Processes all the data from coordinates x1, y1, z1 to x2, y2, z2

        Parameters
        ----------
        x1 : int
            Voxel in the x-axis from where the processing begins
        x2 : int
            Voxel in the x-axis where the processing ends
        y1 : int
            Voxel in the y-axis from where the processing begins
        y2 : int
            Voxel in the y-axis where the processing ends
        z1 : int
            Voxel in the z-axis from where the processing begins
        z2 : int
            Voxel in the z-axis where the processing ends
        args : List
        kwargs : Dict

        Returns
        -------
        MixedProcessor.Results instance
            Object with two properties: correction_parameters and prediction_parameters
        """
        chunks = Chunks(
            self._processor_subjects,
            x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
            mem_usage=self._processor_processing_params['mem_usage']
        )
        dims = chunks.dims

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-3] * dims[-2] * dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Add processing parameters to kwargs
        kwargs['n_jobs'] = self._processor_processing_params['n_jobs']
        kwargs['cache_size'] = self._processor_processing_params['cache_size']

        # Get the results of the first chunk to initialize dimensions of the solution matrices
        # Get first chunk and fit the parameters
        chunk = next(chunks)

        self._processor_fitter.fit(chunk.data, *args, **kwargs)

        # Get the parameters and the dimensions of the solution matrices
        cparams = self._processor_fitter.correction_parameters
        pparams = self._processor_fitter.prediction_parameters
        cpdims = tuple(cparams.shape[:-3] + dims)
        rpdims = tuple(pparams.shape[:-3] + dims)

        # Initialize solution matrices
        correction_parameters = np.zeros(cpdims, dtype=np.float64)
        prediction_parameters = np.zeros(rpdims, dtype=np.float64)

        # Assign first chunk's solutions to solution matrices
        dx, dy, dz = cparams.shape[-3:]
        correction_parameters[:, :dx, :dy, :dz] = cparams
        prediction_parameters[:, :dx, :dy, :dz] = pparams

        # Update progress
        self._processor_update_progress(prog_inc * dx * dy * dz)

        # Now do the same for the rest of the Chunks
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            # Fit the parameters to the data in the chunk
            self._processor_fitter.fit(cdata, *args, **kwargs)

            # Get the optimal parameters and insert them in the solution matrices
            correction_parameters[:, x:x + dx, y:y + dy, z:z + dz] = self._processor_fitter.correction_parameters
            prediction_parameters[:, x:x + dx, y:y + dy, z:z + dz] = self._processor_fitter.prediction_parameters

            # Update progress
            self._processor_update_progress(prog_inc * dx * dy * dz)

        if self.progress != 100.0:
            self._processor_update_progress(10000.0 - self._processor_progress)

        # Call post_processing routine
        return self.__post_process__(prediction_parameters, correction_parameters)


    def curve(self, covariate_parameters, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, t1=None,
              t2=None, tpoints=50, *args, **kwargs):

        """
        Computes tpoints predicted values in the axis of the predictor from t1 to t2 by using the results of
        a previous execution for each voxel in the relative region [x1:x2, y1:y2, z1:z2]. (Only valid for
        one predictor)

        Parameters
        ----------
        covariate_parameters : ndarray
            Prediction parameters obtained for this processor by means of the process() method
        x1 : int
            Voxel in the x-axis from where the curve computation begins
        x2 : int
            Voxel in the x-axis where the curve computation ends
        y1 : int
            Voxel in the y-axis from where the curve computation begins
        y2 : int
            Voxel in the y-axis where the curve computation ends
        z1 : int
            Voxel in the z-axis from where the curve computation begins
        z2 : int
            Voxel in the z-axis where the curve computation ends
        t1 : float
            Value in the axis of the predictor from where the curve computation starts
        t2 : float
            Value in the axis of the predictor from where the curve computation ends
        tpoints : int
            Number of points used to compute the curve, using a linear spacing between t1 and t2

        Returns
        -------
        ndarray
            4D array with the curve values for the tpoints in each voxel from x1, y1, z1 to x2, y2, z2
        """
        if x2 is None:
            x2 = covariate_parameters.shape[-3]
        if y2 is None:
            y2 = covariate_parameters.shape[-2]
        if z2 is None:
            z2 = covariate_parameters.shape[-1]

        if t1 is None:
            t1 = self.predictors.min(axis=0)
        if t2 is None:
            t2 = self.predictors.max(axis=0)



        R = self.predictors.shape[1]
        pparams = covariate_parameters[:, x1:x2, y1:y2, z1:z2]

        if tpoints == -1:
            preds = np.zeros((self.predictors.shape[0], R), dtype=np.float64)
            for i in range(R):
                preds[:, i] = np.sort(self.predictors[:, i])
        else:
            preds = np.zeros((tpoints, R), dtype=np.float64)
            step = (t2 - t1).astype('float') / (tpoints - 1)
            t = t1
            for i in range(tpoints):
                preds[i] = t
                t += step

        return preds.T, self.__curve__(self._processor_fitter, preds, pparams, *args, **kwargs)


    def corrected_values(self, correction_parameters, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, origx=0,
                         origy=0, origz=0, *args, **kwargs):

        """
        Computes the corrected values for the observations with the given fitter and correction parameters

        Parameters
        ----------
        correction_parameters : ndarray
            Array with the computed correction parameters
        x1 : int
            Relative coordinate to origx of the starting voxel in the x-dimension
        x2 : int
            Relative coordinate to origx of the ending voxel in the x-dimension
        y1 : int
            Relative coordinate to origy of the starting voxel in the y-dimension
        y2 : int
            Relative coordinate to origy of the ending voxel in the y-dimension
        z1 : int
            Relative coordinate to origz of the starting voxel in the z-dimension
        z2 : int
            Relative coordinate to origz of the ending voxel in the z-dimension
        origx : int
            Absolute coordinate where the observations start in the x-dimension
        origy : int
            Absolute coordinate where the observations start in the y-dimension
        origz : int
            Absolute coordinate where the observations start in the z-dimension
        args : List
        kwargs : Dictionary

        Notes
        -----
        x1, x2, y1, y2, z1 and z2 are relative coordinates to (origx, origy, origz), being the latter coordinates
        in absolute value (by default, (0, 0, 0)); that is, (origx + x, origy + y, origz + z) is the point to
        which the correction parameters in the voxel (x, y, z) of 'correction_parameters' correspond

        Returns
        -------
        ndarray
            Array with the corrected observations
        """
        if x2 is None:
            x2 = correction_parameters.shape[-3]
        if y2 is None:
            y2 = correction_parameters.shape[-2]
        if z2 is None:
            z2 = correction_parameters.shape[-1]

        correction_parameters = correction_parameters[:, x1:x2, y1:y2, z1:z2]

        x1 += origx
        x2 += origx
        y1 += origy
        y2 += origy
        z1 += origz
        z2 += origz

        chunks = Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                        mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        corrected_data = np.zeros(tuple([chunks.num_subjects]) + dims, dtype=np.float64)

        for chunk in chunks:
            # Get relative (to the solution matrix) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]
            corrected_data[:, x:(x + dx), y:(y + dy), z:(z + dz)] = self.__corrected_values__(
                cdata,
                correction_parameters[:,  x:(x + dx), y:(y + dy), z:(z + dz)],
                *args, **kwargs)

        return corrected_data


    def gm_values(self, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None):
        """
        Returns the original (non-corrected) observations

        Parameters
        ----------
        x1 : int
            Voxel in the x-axis from where the retrieval begins
        x2 : int
            Voxel in the x-axis where the retrieval ends
        y1 : int
            Voxel in the y-axis from where the retrieval begins
        y2 : int
            Voxel in the y-axis where the retrieval ends
        z1 : int
            Voxel in the z-axis from where the retrieval begins
        z2 : int
            Voxel in the z-axis where the retrieval ends

        Returns
        -------
        ndarray
            Array with the original observations
        """
        chunks = Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                        mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        gm_data = np.zeros(tuple([chunks.num_subjects]) + dims, dtype=np.float64)

        for chunk in chunks:
            # Get relative (to the solution matrix) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            gm_data[:, x:(x + dx), y:(y + dy), z:(z + dz)] = cdata

        return gm_data


    def evaluate_fit(self, evaluation_function, correction_parameters, prediction_parameters, x1=0, x2=None, y1=0,
                     y2=None, z1=0, z2=None, origx=0, origy=0, origz=0, gm_threshold=None, filter_nans=True,
                     default_value=0.0, *args, **kwargs):

        """
        Evaluates the goodness of the fit for a particular fit evaluation metric

        Parameters
        ----------
        evaluation_function : FitScores.FitEvaluation function
            Fit evaluation function
        correction_parameters : ndarray
            Array with the correction parameters
        prediction_parameters : ndarray
            Array with the prediction parameters
        x1 : int
            Relative coordinate to origx of the starting voxel in the x-dimension
        x2 : int
            Relative coordinate to origx of the ending voxel in the x-dimension
        y1 : int
            Relative coordinate to origy of the starting voxel in the y-dimension
        y2 : int
            Relative coordinate to origy of the ending voxel in the y-dimension
        z1 : int
            Relative coordinate to origz of the starting voxel in the z-dimension
        z2 : int
            Relative coordinate to origz of the ending voxel in the z-dimension
        origx : int
            Absolute coordinate where the observations start in the x-dimension
        origy : int
            Absolute coordinate where the observations start in the y-dimension
        origz : int
            Absolute coordinate where the observations start in the z-dimension
        gm_threshold : float
            Float that specifies the minimum value of mean gray matter (across subjects) that a voxel must have.
            All voxels that don't fulfill this requirement have their fitting score filtered out
        filter_nans : Boolean
            Whether to filter the values that are not numeric or not
        default_value : float
            Default value for the voxels that have mean gray matter below the threshold or have NaNs in the
            fitting scores
        args : List
        kwargs : Dictionary

        Returns
        -------
        ndarray
            Array with the fitting scores of the specified evaluation function
        """
        # Evaluate fitting from pre-processed parameters
        if correction_parameters.shape[-3] != prediction_parameters.shape[-3] or correction_parameters.shape[-2] != \
                prediction_parameters.shape[-2] or correction_parameters.shape[-1] != prediction_parameters.shape[-1]:
            raise ValueError('The dimensions of the correction parameters and the prediction parameters do not match')

        if x2 is None:
            x2 = x1 + correction_parameters.shape[-3]
        if y2 is None:
            y2 = y1 + correction_parameters.shape[-2]
        if z2 is None:
            z2 = z1 + correction_parameters.shape[-1]

        covariate_parameters = correction_parameters[:, x1:x2, y1:y2, z1:z2]

        x1 += origx
        x2 += origx
        y1 += origy
        y2 += origy
        z1 += origz
        z2 += origz

        chunks = Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                        mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        # Initialize solution matrix
        fitting_scores = np.zeros(dims, dtype=np.float64)

        if gm_threshold is not None:
            # Instead of comparing the mean to the original gm_threshold,
            # we compare the sum to such gm_threshold times the number of subjects
            gm_threshold *= chunks.num_subjects
            invalid_voxels = np.zeros(fitting_scores.shape, dtype=np.bool)

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-3] * dims[-2] * dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Evaluate the fit for each chunk
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            if gm_threshold is not None:
                invalid_voxels[x:(x + dx), y:(y + dy), z:(z + dz)] = np.sum(cdata, axis=0) < gm_threshold

            fitres = FittingResults()

            # Assign the bound data necessary for the evaluation
            bound_functions = self.assign_bound_data(
                observations=cdata,
                correctors=self._processor_fitter.correctors,
                predictors=self._processor_fitter.predictors,
                correction_parameters=correction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)],
                prediction_parameters=prediction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)],
                fitting_results=fitres
            )

            # Bind the functions to the processor instance
            def bind_function(function_name):
                return lambda x: getattr(x.fitting_results, function_name)

            for bound_f in bound_functions:
                eval_func[self].bind(bound_f, bind_function(bound_f))

            # Evaluate the fit for the voxels in this chunk and store them
            fitting_scores[x:x + dx, y:y + dy, z:z + dz] = evaluation_function[self].evaluate(fitres, *args,
                                                                                                   **kwargs)

            # Update progress
            self._processor_update_progress(prog_inc * dx * dy * dz)

        if self.progress != 100.0:
            self._processor_update_progress(10000.0 - self._processor_progress)

        # Filter non-finite elements
        if filter_nans:
            fitting_scores[~np.isfinite(fitting_scores)] = default_value

        # Filter by gray-matter threshold
        if gm_threshold is not None:
            fitting_scores[invalid_voxels] = default_value

        return fitting_scores

    @staticmethod
    def evaluate_latent_space(self, correction_parameters, prediction_parameters, x1=0, x2=None,
                              y1=0, y2=None, z1=0, z2=None, origx=0, origy=0, origz=0,
                              gm_threshold=None, filter_nans=True, default_value=0.0,
                              n_permutations=0, *args, **kwargs):
        """
        Evaluates the goodness of the fit for a particular fit evaluation metric

        Parameters
        ----------
        correction_parameters : ndarray
            Array with the correction parameters
        prediction_parameters : ndarray
            Array with the prediction parameters
        x1 : int
            Relative coordinate to origx of the starting voxel in the x-dimension
        x2 : int
            Relative coordinate to origx of the ending voxel in the x-dimension
        y1 : int
            Relative coordinate to origy of the starting voxel in the y-dimension
        y2 : int
            Relative coordinate to origy of the ending voxel in the y-dimension
        z1 : int
            Relative coordinate to origz of the starting voxel in the z-dimension
        z2 : int
            Relative coordinate to origz of the ending voxel in the z-dimension
        origx : int
            Absolute coordinate where the observations start in the x-dimension
        origy : int
            Absolute coordinate where the observations start in the y-dimension
        origz : int
            Absolute coordinate where the observations start in the z-dimension
        gm_threshold : float
            Float that specifies the minimum value of mean gray matter (across subjects) that a voxel must have.
            All voxels that don't fulfill this requirement have their fitting score filtered out
        filter_nans : Boolean
            Whether to filter the values that are not numeric or not
        default_value : float
            Default value for the voxels that have mean gray matter below the threshold or have NaNs in the
            fitting scores
        args : List
        kwargs : Dictionary

        Returns
        -------
        ndarray
            Array with the fitting scores of the specified evaluation function
        """
        # Evaluate fitting from pre-processed parameters

        if correction_parameters.shape[-3] != prediction_parameters.shape[-3] or correction_parameters.shape[-2] != \
                prediction_parameters.shape[-2] or correction_parameters.shape[-1] != prediction_parameters.shape[-1]:
            raise ValueError('The dimensions of the correction parameters and the prediction parameters do not match')

        if x2 is None:
            x2 = x1 + correction_parameters.shape[-3]
        if y2 is None:
            y2 = y1 + correction_parameters.shape[-2]
        if z2 is None:
            z2 = z1 + correction_parameters.shape[-1]

        correction_parameters = correction_parameters[:, x1:x2, y1:y2, z1:z2]
        prediction_parameters = prediction_parameters[:, x1:x2, y1:y2, z1:z2]

        x1 += origx
        x2 += origx
        y1 += origy
        y2 += origy
        z1 += origz
        z2 += origz

        chunks = Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                        mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        # Initialize solution matrix
        num_latent_components = int(np.unique(np.sort(prediction_parameters[-1]))[-1])
        effect_strength = np.zeros((num_latent_components,) + dims, dtype=np.float64)
        p_value = np.zeros((num_latent_components,) + dims, dtype=np.float64)
        effect_type = np.zeros((num_latent_components, self._processor_fitter.predictors.shape[1])
                               + dims, dtype=np.float64)

        if gm_threshold is not None:
            # Instead of comparing the mean to the original gm_threshold,
            # we compare the sum to such gm_threshold times the number of subjects
            gm_threshold *= chunks.num_subjects
            invalid_voxels = np.zeros(dims, dtype=np.bool)

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-3] * dims[-2] * dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Evaluate the fit for each chunk
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            if gm_threshold is not None:
                invalid_voxels[x:(x + dx), y:(y + dy), z:(z + dz)] = np.sum(cdata, axis=0) < gm_threshold

            fitres = FittingResults()

            # Assign the bound data necessary for the evaluation
            bound_functions = self.assign_bound_data(
                observations=cdata,
                correctors=self._processor_fitter.correctors,
                predictors=self._processor_fitter.predictors,
                correction_parameters=correction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)],
                prediction_parameters=prediction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)],
                fitting_results=fitres
            )

            # Bind the functions to the processor instance
            def bind_function(function_name):
                return lambda x: getattr(x.fitting_results, function_name)

            for bound_f in bound_functions:
                eval_func[self].bind(bound_f, bind_function(bound_f))

            # Evaluate the fit for the voxels in this chunk and store them

            effect_strength[:, x:x + dx, y:y + dy, z:z + dz] = eff_size_eval[self].evaluate(fitres, *args,
                                                                                                 **kwargs)
            p_value[:, x:x + dx, y:y + dy, z:z + dz] = 0
            # p_value[:, x:x + dx, y:y + dy, z:z + dz] = \
            #     eff_size_value_eval[processor].evaluate(fitres,
            #                                             hyp_value = effect_strength[:, x:x + dx, y:y + dy, z:z + dz],
            #                                             num_permutations = n_permutations, *args, **kwargs)
            effect_type[:, :, x:x + dx, y:y + dy, z:z + dz] = eff_type_eval[self].evaluate(fitres, *args, **kwargs)

            # Update progress
            self._processor_update_progress(prog_inc * dx * dy * dz)

        if self.progress != 100.0:
            self._processor_update_progress(10000.0 - self._processor_progress)

        # Filter non-finite elements
        if filter_nans:
            effect_strength[~np.isfinite(effect_strength)] = default_value
            p_value[~np.isfinite(effect_strength)] = 1

        # Filter by gray-matter threshold
        if gm_threshold is not None:
            effect_strength[:, invalid_voxels] = default_value
            p_value[:, invalid_voxels] = 1
            effect_type[:, :, invalid_voxels] = default_value

        return (effect_strength, p_value, effect_type)


class MixedSurfaceProcessor(MixedProcessor):


    def process(self, x1=0, x2=None, *args, **kwargs):
        """
        Processes all the data from coordinates x1, y1, z1 to x2, y2, z2

        Parameters
        ----------
        x1 : int
            Voxel in the x-axis from where the processing begins
        x2 : int
            Voxel in the x-axis where the processing ends
        y1 : int
            Voxel in the y-axis from where the processing begins
        y2 : int
            Voxel in the y-axis where the processing ends
        z1 : int
            Voxel in the z-axis from where the processing begins
        z2 : int
            Voxel in the z-axis where the processing ends
        args : List
        kwargs : Dict

        Returns
        -------
        MixedProcessor.Results instance
            Object with two properties: correction_parameters and prediction_parameters
        """
        chunks = Chunks(
            self._processor_subjects, x1=x1, x2=x2,
            mem_usage=self._processor_processing_params['mem_usage']
        )
        dims = chunks.dims

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Add processing parameters to kwargs
        kwargs['n_jobs'] = self._processor_processing_params['n_jobs']
        kwargs['cache_size'] = self._processor_processing_params['cache_size']

        # Get the results of the first chunk to initialize dimensions of the solution matrices
        # Get first chunk and fit the parameters
        chunk = next(chunks)

        self._processor_fitter.fit(chunk.data, *args, **kwargs)

        # Get the parameters and the dimensions of the solution matrices
        cparams = self._processor_fitter.correction_parameters
        pparams = self._processor_fitter.prediction_parameters
        cpdims = tuple(cparams.shape[:-1] + dims)
        rpdims = tuple(pparams.shape[:-1] + dims)

        # Initialize solution matrices
        correction_parameters = np.zeros(cpdims, dtype=np.float64)
        prediction_parameters = np.zeros(rpdims, dtype=np.float64)

        # Assign first chunk's solutions to solution matrices
        dx = cparams.shape[-1]
        correction_parameters[:, :dx] = cparams
        prediction_parameters[:, :dx] = pparams

        # Update progress
        self._processor_update_progress(prog_inc * dx)

        # Now do the same for the rest of the Chunks
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x = chunk.coords
            x -= x1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx = cdata.shape[-1]

            # Fit the parameters to the data in the chunk
            self._processor_fitter.fit(cdata, *args, **kwargs)

            # Get the optimal parameters and insert them in the solution matrices
            correction_parameters[:, x:x + dx] = self._processor_fitter.correction_parameters
            prediction_parameters[:, x:x + dx] = self._processor_fitter.prediction_parameters

            # Update progress
            self._processor_update_progress(prog_inc * dx)

        if self.progress != 100.0:
            self._processor_update_progress(10000.0 - self._processor_progress)

        # Call post_processing routine
        return self.__post_process__(prediction_parameters, correction_parameters)


    def curve(self, covariate_parameters, x1=0, x2=None, t1=None, t2=None, tpoints=50, *args, **kwargs):
        """
            Computes tpoints predicted values in the axis of the predictor from t1 to t2 by using the results of
            a previous execution for each voxel in the relative region [x1:x2, y1:y2, z1:z2]. (Only valid for
            one predictor)

            Parameters
            ----------
            covariate_parameters : ndarray
                Prediction parameters obtained for this processor by means of the process() method
            x1 : int
                Voxel in the x-axis from where the curve computation begins
            x2 : int
                Voxel in the x-axis where the curve computation ends
            y1 : int
                Voxel in the y-axis from where the curve computation begins
            y2 : int
                Voxel in the y-axis where the curve computation ends
            z1 : int
                Voxel in the z-axis from where the curve computation begins
            z2 : int
                Voxel in the z-axis where the curve computation ends
            t1 : float
                Value in the axis of the predictor from where the curve computation starts
            t2 : float
                Value in the axis of the predictor from where the curve computation ends
            tpoints : int
                Number of points used to compute the curve, using a linear spacing between t1 and t2

            Returns
            -------
            ndarray
                4D array with the curve values for the tpoints in each voxel from x1, y1, z1 to x2, y2, z2
            """
        if x2 is None:
            x2 = covariate_parameters.shape[-1]

        if t1 is None:
            t1 = self.predictors.min(axis=0)
        if t2 is None:
            t2 = self.predictors.max(axis=0)

        R = self.predictors.shape[1]
        pparams = covariate_parameters[:, x1:x2]

        if tpoints == -1:
            preds = np.zeros((self.predictors.shape[0], R), dtype=np.float64)
            for i in range(R):
                preds[:, i] = np.sort(self.predictors[:, i])
        else:
            preds = np.zeros((tpoints, R), dtype=np.float64)
            step = (t2 - t1).astype('float') / (tpoints - 1)
            t = t1
            for i in range(tpoints):
                preds[i] = t
                t += step

        return preds.T, self.__curve__(self._processor_fitter, preds, pparams, *args, **kwargs)


    def corrected_values(self, correction_parameters, x1=0, x2=None, origx=0, *args, **kwargs):
        """
            Computes the corrected values for the observations with the given fitter and correction parameters

            Parameters
            ----------
            correction_parameters : ndarray
                Array with the computed correction parameters
            x1 : int
                Relative coordinate to origx of the starting voxel in the x-dimension
            x2 : int
                Relative coordinate to origx of the ending voxel in the x-dimension
            origx : int
                Absolute coordinate where the observations start in the x-dimension
            args : List
            kwargs : Dictionary

            Notes
            -----
            x1, x2, y1, y2, z1 and z2 are relative coordinates to (origx, origy, origz), being the latter coordinates
            in absolute value (by default, (0, 0, 0)); that is, (origx + x, origy + y, origz + z) is the point to
            which the correction parameters in the voxel (x, y, z) of 'correction_parameters' correspond

            Returns
            -------
            ndarray
                Array with the corrected observations
            """
        if x2 is None:
            x2 = correction_parameters.shape[-1]

        correction_parameters = correction_parameters[:, x1:x2]

        x1 += origx
        x2 += origx

        chunks = Chunks(self._processor_subjects, x1=x1, x2=x2,
                        mem_usage=self._processor_processing_params['mem_usage'])

        dims = chunks.dims
        corrected_data = np.zeros(tuple([chunks.num_subjects]) + dims, dtype=np.float64)

        for chunk in chunks:
            # Get relative (to the solution matrix) coordinates of the chunk
            x = chunk.coords
            x -= x1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx = cdata.shape[-1]

            corrected_data[:, x:(x + dx)] = self.__corrected_values__(cdata,
                                                                      correction_parameters[:, x:(x + dx)],
                                                                      *args, **kwargs)
        return corrected_data


    def gm_values(self, x1=0, x2=None):
        """
            Returns the original (non-corrected) observations

            Parameters
            ----------
            x1 : int
                Voxel in the x-axis from where the retrieval begins
            x2 : int
                Voxel in the x-axis where the retrieval ends

            Returns
            -------
            ndarray
                Array with the original observations
            """
        chunks = Chunks(self._processor_subjects, x1=x1,
                        mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        gm_data = np.zeros(tuple([chunks.num_subjects]) + dims, dtype=np.float64)

        for chunk in chunks:
            # Get relative (to the solution matrix) coordinates of the chunk
            x = chunk.coords
            x -= x1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx = cdata.shape[-1:]

            gm_data[:, x:(x + dx)] = cdata

        return gm_data


    def evaluate_fit(self, evaluation_function, correction_parameters, prediction_parameters, x1=0, x2=None, origx=0,
                     gm_threshold=None, filter_nans=True, default_value=0.0, *args, **kwargs):

        """
        Evaluates the goodness of the fit for a particular fit evaluation metric

        Parameters
        ----------
        evaluation_function : FitScores.FitEvaluation function
            Fit evaluation function
        covariate_parameters : ndarray
            Array with the correction parameters
        x1 : int
            Relative coordinate to origx of the starting voxel in the x-dimension
        x2 : int
            Relative coordinate to origx of the ending voxel in the x-dimension
        origx : int
            Absolute coordinate where the observations start in the x-dimension
        gm_threshold : float
            Float that specifies the minimum value of mean gray matter (across subjects) that a voxel must have.
            All voxels that don't fulfill this requirement have their fitting score filtered out
        filter_nans : Boolean
            Whether to filter the values that are not numeric or not
        default_value : float
            Default value for the voxels that have mean gray matter below the threshold or have NaNs in the
            fitting scores
        args : List
        kwargs : Dictionary

        Returns
        -------
        ndarray
            Array with the fitting scores of the specified evaluation function
            """
        # Evaluate fitting from pre-processed parameters
        if correction_parameters.shape[-1] != prediction_parameters.shape[-1]:
            raise ValueError(
                'The dimensions of the correction parameters and the prediction parameters do not match')

        kwargs = {key: value for key, value in kwargs.items() if key not in ['y1','y2','z1','z2']}
        if x2 is None:
            x2 = x1 + correction_parameters.shape[-1]

        x1 += origx
        x2 += origx


        chunks = Chunks(self._processor_subjects, x1=x1,
                        mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        # Initialize solution matrix
        fitting_scores = np.zeros(dims, dtype=np.float64)

        if gm_threshold is not None:
            # Instead of comparing the mean to the original gm_threshold,
            # we compare the sum to such gm_threshold times the number of subjects
            gm_threshold *= chunks.num_subjects
            invalid_voxels = np.zeros(fitting_scores.shape, dtype=np.bool)

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Evaluate the fit for each chunk
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x = chunk.coords
            x -= x1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx = cdata.shape[-1]

            if gm_threshold is not None:
                invalid_voxels[x:(x + dx)] = np.sum(cdata, axis=0) < gm_threshold

            fitres = FittingResults()

            # Assign the bound data necessary for the evaluation
            bound_functions = self.assign_bound_data(
                observations=cdata,
                correctors=self._processor_fitter.correctors,
                predictors=self._processor_fitter.predictors,
                correction_parameters=correction_parameters[:, x:(x + dx)],
                prediction_parameters=prediction_parameters[:, x:(x + dx)],
                fitting_results=fitres
            )

            # Bind the functions to the processor instance
            def bind_function(function_name):
                return lambda x: getattr(x.fitting_results, function_name)

            for bound_f in bound_functions:
                eval_func[self].bind(bound_f, bind_function(bound_f))

            # Evaluate the fit for the voxels in this chunk and store them
            fitting_scores[x:x + dx] = evaluation_function[self].evaluate(fitres, *args, **kwargs)

            # Update progress
            self._processor_update_progress(prog_inc * dx)

        if self.progress != 100.0:
            self._processor_update_progress(10000.0 - self._processor_progress)

        # Filter non-finite elements
        if filter_nans:
            fitting_scores[~np.isfinite(fitting_scores)] = default_value

        # Filter by gray-matter threshold
        if gm_threshold is not None:
            fitting_scores[invalid_voxels] = default_value
            print('The following proportion of vertices is not taken into account due to gm_threshold: ' +
                  str(len(np.where(invalid_voxels == True)[0]) / np.prod(dims)))

        return fitting_scores


    def evaluate_latent_space(self, correction_parameters, prediction_parameters, x1=0, x2=None, origx=0,
                              gm_threshold=None, filter_nans=True, default_value=0.0, n_permutations=0,
                              *args, **kwargs):

        raise ValueError('SurfaceProcessor not yet implemented')



MixedProcessor._mixedprocessor_perp_norm_options = {MixedProcessor._mixedprocessor_perp_norm_options_names[i]: i for i
                                                    in
                                                    range(len(MixedProcessor._mixedprocessor_perp_norm_options_names))}


