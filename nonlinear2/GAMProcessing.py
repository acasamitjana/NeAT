from Processing import Processor
from GAM import GAM, InterceptSmoother, PolynomialSmoother, SplinesSmoother, SmootherSet
import numpy as np


class GAMProcessor(Processor):
	_gamprocessor_perp_norm_options_names = [
		'Orthonormalize all',
		'Orthogonalize all',
		'Normalize all',
		'Orthonormalize predictors',
		'Orthogonalize predictors',
		'Normalize predictors',
		'Orthonormalize correctors',
		'Orthogonalize correctors',
		'Normalize correctors',
		'Use correctors and predictors as they are'
	]

	_gamprocessor_perp_norm_options_list = [
		GAM.orthonormalize_all,
		GAM.orthogonalize_all,
		GAM.normalize_all,
		GAM.orthonormalize_predictors,
		GAM.orthogonalize_predictors,
		GAM.normalize_predictors,
		GAM.orthonormalize_correctors,
		GAM.orthogonalize_correctors,
		GAM.normalize_correctors,
		lambda *args, **kwargs: None
	]

	TYPE_SMOOTHER=[InterceptSmoother,PolynomialSmoother,SplinesSmoother]

	def __fitter__(self, user_defined_parameters):
		'''Initializes the GAM fitter to be used to process the data.
		'''

		class newgam(GAM):

			@property
			def regression_parameters(self):
				''' Implements an add-on that allows regression parameters to have flexible dimensions
				'''
				pparams = super(newgam, self).regression_parameters
				res = np.zeros((277,) + pparams.shape[1:])
				res[:pparams.shape[0]] = pparams
				return res

		self._gamprocessor_perp_norm_option = user_defined_parameters[0]
		self._gamprocessor_smoother_parameters = user_defined_parameters[1]

		sm_index = 0
		corrector_smoothers=SmootherSet()
		predictor_smoothers=SmootherSet()
		for corr in self.correctors.T:
			smoother_function=GAMProcessor.TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](corr)
			sm_index += 1
			n_param = self._gamprocessor_smoother_parameters[sm_index]
			sm_index += 1
			smoother_function.set_parameters(np.array(self._gamprocessor_smoother_parameters[sm_index:sm_index+n_param])[:,None])
			sm_index += n_param
			corrector_smoothers.extend(smoother_function)
		for reg in self.predictors.T:
			smoother_function=GAMProcessor.TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](reg)
			sm_index += 1
			n_param = self._gamprocessor_smoother_parameters[sm_index]
			sm_index += 1
			smoother_function.set_parameters(np.array(self._gamprocessor_smoother_parameters[sm_index:sm_index+n_param])[:,None])
			sm_index += n_param
			predictor_smoothers.extend(smoother_function)

		treat_data = GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option]

		gam = GAM(corrector_smoothers=corrector_smoothers, predictor_smoothers=predictor_smoothers)

		treat_data(gam)

		return gam

	# def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None, evaluation_kwargs = {}, *args, **kwargs):
	# 	results = super(self, GAMProcessor).process(x1 = x1, x2 = x2, y1 = y1, y2 = y2, z1 = z1, z2 = z2,mem_usage = mem_usage, evaluation_kwargs = evaluation_kwargs, *args, **kwargs)
	# 	Mirar cual es el maximo numero de parametros en la matriz results.prediction_parameters
	# 	Cortar matriz new_regression_parameters = results.regression_parameters[:max_num_params]
	# 	return self.Results(new_regression_parameters, results.correction_parameters, results.fitting_scores)

	def __user_defined_parameters__(self, fitter):
		return (self._gamprocessor_perp_norm_option,self._gamprocessor_smoother_parameters)

	def __read_user_defined_parameters__(self, predictor_names, corrector_names):

		perp_norm_option = GAMProcessor._gamprocessor_perp_norm_options[super(GAMProcessor, self).__getoneof__(
			GAMProcessor._gamprocessor_perp_norm_options_names,
			default_value = 'Orthonormalize all',
			show_text = 'GAM Processor: How do you want to treat the features? (default: Orthonormalize all)'
		)]

		smoothing_functions = []
		print('')
		for cor in corrector_names:
			smoother_type = super(GAMProcessor, self).__getint__(
				default_value = 1,
				try_ntimes = 3,
				show_text = 'GAM Processor: Please, enter the smoothing function of the feature (corrector) \'' + str(cor)
							+ '\' (1: Polynomial Smoother, 2: Splines Smoother): ')
			smoothing_functions.append(smoother_type)

			if smoother_type == GAMProcessor.TYPE_SMOOTHER.index(PolynomialSmoother):
				smoothing_functions.append(1)
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 3,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
								'(or leave blank to set to 3) '
				))
			elif smoother_type == GAMProcessor.TYPE_SMOOTHER.index(SplinesSmoother):
				smoothing_functions.append(2)
				smoothing_functions.append(super(GAMProcessor, self).__getfloat__(
					default_value = 500,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the smoothing factor of the spline'
								'(or leave it blank to set it to default: 500) '
				))
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 3,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
								'(or leave blank to set to 3) '
				))

		for reg in predictor_names:
			smoother_type = super(GAMProcessor, self).__getint__(
				default_value = 1,
				try_ntimes = 3,
				show_text = 'GAM Processor: Please, enter the smoothing function of the feature (predictor) \'' + str(reg)
							+ '\' (1: Polynomial Smoother, 2: Splines Smoother): ')
			smoothing_functions.append(smoother_type)

			if smoother_type == GAMProcessor.TYPE_SMOOTHER.index(PolynomialSmoother):
				smoothing_functions.append(1)
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 3,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
								'(or leave blank to set to 3) '
				))
			elif smoother_type == GAMProcessor.TYPE_SMOOTHER.index(SplinesSmoother):
				smoothing_functions.append(2)
				smoothing_functions.append(super(GAMProcessor, self).__getfloat__(
					default_value = 500,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the smoothing factor of the splines'
								'(or leave it blank to set it to default: 500) '
				))
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 3,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
								'(or leave blank to set to 3) '
				))


		return (perp_norm_option,smoothing_functions)

	def __curve__(self, fitter, regressors, regression_parameters):
		gam = GAM()
		GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option](gam)
		return gam.predict(regressors=regressors, regression_parameters = regression_parameters)


GAMProcessor._gamprocessor_perp_norm_options = {GAMProcessor._gamprocessor_perp_norm_options_names[i] : i for i in xrange(len(GAMProcessor._gamprocessor_perp_norm_options_names))}

