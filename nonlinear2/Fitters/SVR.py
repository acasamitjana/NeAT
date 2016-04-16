""" Support Vector Regression fitters
        * Linear SVR
        * Polynomial SVR
        * Gaussian SVR
"""
import numpy as np
import sklearn.preprocessing as preprocessing
from joblib import Parallel, delayed
from numpy import array, ravel, zeros
from sklearn.svm import LinearSVR

from nonlinear2.Fitters.CurveFitting import AdditiveCurveFitter
from nonlinear2.Utils.Transforms import polynomial


class LinSVR(AdditiveCurveFitter):
    """
    LINEAR SVR
    Class that implements linear Support Vector Regression
    """

    def __init__(self, predictors = None, correctors = None, homogeneous = True):
        self._svr_homogeneous = homogeneous
        # Don't allow a homogeneous feature to be created, use instead the intercept term from the fitter
        super(LinSVR, self).__init__(predictors, correctors, homogeneous)

    def __fit__(self, correctors, predictors, observations, *args, **kwargs):
        """

        Parameters
        ----------
        correctors
        predictors
        observations
        args
        kwargs

        Returns
        -------

        """
        # Parameters for linear SVR
        C = kwargs['C'] if 'C' in kwargs else 100.0
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.01
        shrinking = kwargs['shrinking'] if 'shrinking' in kwargs else True
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1000
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-3
        sample_weight = kwargs['sample_weight'] if 'sample_weight' in kwargs else None
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 4

        # Initialize linear SVR from scikit-learn

        svr_fitter = LinearSVR(epsilon=epsilon, tol=tol, C=C, fit_intercept=self._svr_homogeneous,
                               intercept_scaling=C, max_iter=max_iter)

        num_variables = observations.shape[1]

        # Fit predictors
        predictors_std = preprocessing.scale(predictors)
        params = Parallel(n_jobs=n_jobs)(delayed(__fit_features__) \
                                        (svr_fitter, predictors_std, observations[:, i], sample_weight, False)
                                         for i in range(num_variables))
        pparams = array(params).T

        # Fit correctors
        if self._svr_homogeneous:
            correctors = correctors[:, 1:]
        c_size = correctors.size

        if c_size != 0:
            correctors_std = preprocessing.scale(correctors)
        elif c_size == 0 and self._svr_homogeneous:
            correctors_std = np.array([[]])
        else:
            cparams = np.array([[]])
            return cparams, pparams

        params = Parallel(n_jobs=n_jobs)(delayed(__fit_features__) \
                                    (svr_fitter, correctors_std, observations[:, i], sample_weight, self._svr_homogeneous)
                                     for i in range(num_variables))

        cparams = array(params).T

        # Get correction and regression coefficients
        return cparams, pparams

    @staticmethod
    def __predict__(predictors, regression_parameters, *args, **kwargs):
        """

        Parameters
        ----------
        predictors
        regression_parameters
        args
        kwargs

        Returns
        -------

        """
        return predictors.dot(regression_parameters)

class PolySVR(LinSVR):
    """ POLYNOMIAL SVR """

    def __init__(self, features, predictors = None, degrees = None, homogeneous = True):
        """

        Parameters
        ----------
        features NxF (2-dimensional) matrix
        predictors int / iterable object (default None)
        degrees iterable of F elements (default None)
        homogeneous bool (default True)

        Returns
        -------

        """

        # Check features matrix
        self._svr_features = array(features)
        if len(self._svr_features.shape) != 2:
            raise ValueError('Argument \'features\' must be a 2-dimensional matrix')
        self._svr_features = self._svr_features.T

        # Check predictors indexes
        if predictors is None:
            self._svr_is_predictor = [True]*len(self._svr_features)
            predictors = []
        else:
            self._svr_is_predictor = [False]*len(self._svr_features)
            if isinstance(predictors, int):
                predictors = [predictors]
        try:
            for r in predictors:
                try:
                    self._svr_is_predictor[r] = True
                except TypeError:
                    raise ValueError('All elements in argument \'predictors\' must be valid indices')
                except IndexError:
                    raise IndexError('Index out of range in argument \'predictors\'')
        except TypeError:
            raise TypeError('Argument \'predictors\' must be iterable or int')

        # Check degrees indexes
        if degrees is None:
            self._svr_degrees = [1]*len(self._svr_features)
        else:
            degrees = list(degrees)
            if len(degrees) != len(self._svr_features):
                raise ValueError('Argument \'degrees\' must have a length equal to the number of features')
            for deg in degrees:
                if not isinstance(deg, int):
                    raise ValueError('Expected integer in \'degrees\' list, got ' + str(type(deg)) + ' instead')
                if deg < 1:
                    raise ValueError('All degrees must be >= 1')
            self._svr_degrees = degrees

        # Check homogeneous term
        self._svr_homogeneous = homogeneous

        # Call function to expand the feature space with polynomial terms
        self.__svr_polynomial_expansion()

    def __svr_polynomial_expansion(self):
        """
        Expands the input space to a feature space with the corresponding polynomial terms,
        and then uses this expanded space to initialize the correctors and predictors for a linear SVR
        Returns
        -------

        """
        correctors = []
        predictors = []
        for index in xrange(len(self._svr_is_predictor)):
            for p in polynomial(self._svr_degrees[index], [self._svr_features[index]]):
                if self._svr_is_predictor[index]:
                    predictors.append(p)
                else:
                    correctors.append(p)

        if len(correctors) == 0:
            correctors = None
        else:
            correctors = array(correctors).T

        if len(predictors) == 0:
            predictors = None
        else:
            predictors = array(predictors).T

        # Instance a LinSVR (parent) with the expanded polynomial features
        super(PolySVR, self).__init__(predictors, correctors, self._svr_homogeneous)


class GaussianSVR(object):
    """ GAUSSIAN SVR """
    pass


""" HELPER FUNCTIONS """

def __fit_features__(fitter, X, y, sample_weight=None, homogeneous=True):
        """
        Fits the features from X to the observation y given the linear fitter and the optional sample_weights
        Parameters
        ----------
        fitter:  sklearn linear fitter, must have the fit method and the coef_ and intercept_ attributes
        X:   NxF matrix, where N is the number of observations and F the number of features
        y:   Nx1 the variable that we want to explain with the features
        [sample_weight]:    array with weights assigned to each feature
        [homogeneous]: Boolean whether the intercept term must be calculated

        Returns
        -------
        {numpy array} (F+1)x1 array with the fitting coefficients and the intercept term if homogeneous=True,
        otherwise Fx1 array with only the fitting coefficients
        """
        num_features = X.shape[1]
        if num_features <= 0:
            if homogeneous:
                # If the features array is empty and we need to compute the homogeneous term,
                # create dummy features to fit and then get only the intercept term
                X = np.ones((y.shape[0], 1))
            else:
                raise Exception("Features array X is not a NxF array")
        fitter.fit(X, y)

        if homogeneous:
            if num_features > 0:
                params = zeros((num_features + 1, 1))
                coefficients = np.atleast_2d(fitter.coef_)
                params[1:, :] = coefficients.T
            else:
                params = zeros((1, 1)) # Only the intercept term
            params[0, :] = float(fitter.intercept_)
        else:
            params = fitter.coef_.T
        return ravel(params)