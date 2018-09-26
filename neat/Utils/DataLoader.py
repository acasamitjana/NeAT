import nibabel as nib
import numpy as np
import yaml

from os.path import join, basename
from neat.Utils.Subject import Subject
from neat.Utils.ExcelIO import ExcelSheet, CSVSheet
from neat.Utils.niftiIO import file_reader_from_extension, file_writer_from_extension, Results, NiftiReader


class DataLoader(object):
    """
    Loads the subjects and the configuration of a study given the path to the configuration file for this study
    """

    def __init__(self, configuration_path):
        """
        Initializes a DataLoader with the given configuration file

        Parameters
        ----------
        configuration_path : String
            Path to the YAMP configuration file with the configuration parameters expected for a study.
            See config/exampleConfig.yaml for more information about the format of configuration files.
        """
        # Load the configuration
        with open(configuration_path, 'r') as conf_file:
            conf = yaml.load(conf_file)
        self._conf = conf
        self._cached_subjects = []
        self._start = None
        self._end = None

    def get_subjects(self, start=None, end=None):
        """
        Gets all the subjects from the study given the configuration parameters of this instance of DataLoader

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located

        Returns
        -------
        list<Subject>
            List of all subjects

        Raises
        ------
        KeyError
            If the configuration file doesn't follow the format rules or if one or more identifiers used in
            the configuration file don't exist.
        """

        # Load model parameters from configuration
        excel_file = self._conf['input']['excel_file']
        data_folder = self._conf['input']['data_folder']
        study_prefix = self._conf['input']['study_prefix']
        extension = self._conf['input']['extension']



        # Load model parameters
        id_identifier = self._conf['model']['id_identifier']  # ID identifier
        id_type = int if self._conf['model']['id_type'] == 'Number' else str
        category_identifier = self._conf['model']['category_identifier']  # Category identifier
        fields_names = []
        if self._conf['model']['correctors_identifiers'] is not None:
            fields_names = fields_names + list(self._conf['model']['correctors_identifiers'])  # Correctors
        if self._conf['model']['predictor_identifier'] is not None:
            fields_names = fields_names + list(self._conf['model']['predictor_identifier'])        # Predictors

        # Load excel file
        if excel_file.endswith('.xls'):
            xls = ExcelSheet(excel_file)
        elif excel_file.endswith('.csv'):
            xls = CSVSheet(excel_file)
        else:
            raise ValueError('Specify one of the valid formats for excel file [xls,csv]')


        # Prepare fields type parameter
        if category_identifier:
            # If there is a category identifier, add the id identifier and the category identifier
            fields = {
                id_identifier: id_type,
                category_identifier: int
            }
        else:
            # Otherwise, just add the id identifier
            fields = {
                id_identifier: id_type
            }
        for field in fields_names:
            fields[field] = float

        # Load the predictors and correctors for all subjects
        subjects = []
        end = end if end is None else end + 1
        for row in xls.get_rows(start=start, end=end, fieldstype=fields):
            # The subjects must have a non-empty ID
            if row[id_identifier] != "":
                # Create path to nifti file
                nifti_path = join(data_folder, study_prefix + str(row[id_identifier]) + extension)
                # Category
                category = row[category_identifier] if category_identifier else None
                # Create subject
                subj = Subject(row[id_identifier], nifti_path, extension, category=category)
                # Add prediction and correction parameters
                for param_name in fields_names:
                    subj.set_parameter(parameter_name=param_name, parameter_value=row[param_name])
                # Append subject to the subjects' list
                subjects.append(subj)

        # Cache subjects
        self._cached_subjects = subjects
        self._start = start
        self._end = end
        # Return the cached subjects
        return self._cached_subjects

    def get_template(self):
        """
        Returns the template specified in the configuration file

        Returns
        -------
        ndarray
            3D numpy array that contains the template image
        """
        template_path = self._conf['input']['template_file']
        reader = file_reader_from_extension(basename(template_path))
        return reader(template_path).get_data()

    def get_template_affine(self):
        """
        Returns the affine matrix used to map between the template coordinates space and the voxel coordinates space

        Returns
        -------
        numpy.array
            The affine matrix in the template NIFTI file
        """
        template_path = self._conf['input']['template_file']
        extension = self._conf['input']['extension']
        reader = file_reader_from_extension(extension)

        if reader == NiftiReader:
            return reader(template_path).affine()
        else:
            return None

    def get_gm_data(self, start=None, end=None, use_cache=True):
        """
        Returns the grey-matter data of all subjects between "start" and "end" indices [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            4D matrix with the grey matter values of all voxels for all subjects
        """
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        gm_values = list(map(lambda subject: nib.load(subject.gmfile).get_data(), subjects))
        return np.asarray(gm_values)

    def get_predictor(self, start=None, end=None, use_cache=True):
        """
        Returns the predictors of the study for all the subjects between "start" and "end" [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            2D matrix with the predictors for all subjects
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        if self._conf['model']['predictor_identifier'] is not None:
            predictors_names = self._conf['model']['predictor_identifier']
            predictors = list(map(lambda subject: subject.get_parameters(predictors_names), subjects))
        else:
            predictors = []

        return np.asarray(predictors)

    def get_correctors(self, start=None, end=None, use_cache=True):
        """
        Returns the correctors of the study for all the subjects between "start" and "end" [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            2D matrix with the correctors for all subjects
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        # correctors = [0]*len(subjects)
        if self._conf['model']['correctors_identifiers'] is not None:
            correctors_names = self._conf['model']['correctors_identifiers']
            correctors = list(map(lambda subject: subject.get_parameters(correctors_names), subjects))
        else:
            correctors = np.zeros((len(subjects),1))

        return np.asarray(correctors)

    def get_predictor_name(self):
        """
        Returns the names of the predictors of this study

        Returns
        -------
        List<String>
            List of predictors' names
        """
        if self._conf['model']['predictor_identifier'] is not None:
            return self._conf['model']['predictor_identifier']
        else:
            return []

    def get_correctors_names(self):
        """
        Returns the correctors' names of this study

        Returns
        -------
        List<String>
            List of correctors' names
        """
        if self._conf['model']['correctors_identifiers'] is not None:
            return self._conf['model']['correctors_identifiers']
        else:
            return ['all-zeros']

    def get_processing_parameters(self):
        """
        Returns the parameters used for processing, that is, number of jobs, chunk memory, and cache size

        Returns
        -------
        dict
            Dictionary with keys 'n_jobs', 'mem_usage' and 'cache_size' representing
            number of jobs used for fitting, amount of memory in MB per chunck, and amount of memory
            reserved for SVR fitting, respectively.
        """

        return self._conf['processing_params']

    def get_output_dir(self):
        """
        Returns the path to the output folder set in the configuration file

        Returns
        -------
        String
            Path to the output folder
        """
        return self._conf['output']['output_path']

    def get_hyperparams_finding_configuration(self, fitting_method='PolySVR'):
        """
        Returns a GridSearch ready dictionary with the possible values for the specified hyperparameters

        Returns
        -------
        Dictionary
            The keys of the dictionary are the name of the hyperparameter and the values the numpy array
            containing all the possible values amongst which the optimal will be found.
        """

        # Inner function
        def get_hyperparams(hyperparams_dict, hyperparam_name):

            identifier = '{}_values'.format(hyperparam_name)

            start_val = hyperparams_config[identifier]['start']
            end_val = hyperparams_config[identifier]['end']
            N = hyperparams_config[identifier]['N']

            if hyperparams_config[identifier]['spacing'] == 'logarithmic':
                if hyperparams_config[identifier]['method'] == 'random':
                    # Logarithmic spacing and random search
                    hyperparams_dict[hyperparam_name] = np.sort([10 ** i for i in np.random.uniform(
                        start_val, end_val, N
                    )])
                else:
                    # Logarithmic spacing and deterministic search
                    hyperparams_dict[hyperparam_name] = np.logspace(start_val, end_val, N)
            else:
                if hyperparams_config[identifier]['method'] == 'random':
                    # Linear spacing and random search
                    hyperparams_dict[hyperparam_name] = np.sort(np.random.uniform(
                        start_val, end_val, N
                    ))
                else:
                    # Logarithmic spacing and deterministic search
                    hyperparams_dict[hyperparam_name] = np.linspace(start_val, end_val, N)

        hyperparams_config = self._conf['hyperparameters_finding']
        hyperparams_dict = {}
        if hyperparams_config['epsilon']:
            get_hyperparams(hyperparams_dict, 'epsilon')
        if hyperparams_config['C']:
            get_hyperparams(hyperparams_dict, 'C')
        if fitting_method == 'GaussianSVR':
            if hyperparams_config['gamma']:
                get_hyperparams(hyperparams_dict, 'gamma')

        return hyperparams_dict

    def get_results_io(self):
        """
        Returns the class in charge of results I/O either for loading/writing results.
        Returns
        -------
        niftiIO.Results class with two properties:
            loader = instance class for loading results
            writer = instance class for writing results

        """

        extension = self._conf['output']['extension']
        loader = file_reader_from_extension(extension)
        writer = file_writer_from_extension(extension)

        return Results(loader, writer, extension)

    def get_extension(self):
        """
        Returns the extention of the file

        Returns
        -------
        extension: string


        """
        return self._conf['input']['extension']

    def get_prefix(self):
        return self._conf['input']['study_prefix']


class DataLoaderLongitudinal(DataLoader):

    def get_subjects(self, start=None, end=None):
        """
        Gets all the subjects from the study given the configuration parameters of this instance of DataLoader

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located

        Returns
        -------
        list<Subject>
            List of all subjects

        Raises
        ------
        KeyError
            If the configuration file doesn't follow the format rules or if one or more identifiers used in
            the configuration file don't exist.
        """

        # Load model parameters from configuration
        excel_file = self._conf['input']['excel_file']
        data_folder = self._conf['input']['data_folder']
        study_prefix = self._conf['input']['study_prefix']
        extension = self._conf['input']['extension']



        # Load model parameters
        subject_identifier = self._conf['model']['subject_identifier']  # ID identifier
        subject_id_type = int if self._conf['model']['subject_id_type'] == 'Number' else str
        event_identifier = self._conf['model']['event_identifier']  # ID identifier
        event_id_type = int if self._conf['model']['event_id_type'] == 'Number' else str
        category_identifier = self._conf['model']['category_identifier']  # Category identifier
        dt_identifier = self._conf['model']['dt_identifier']  # Category identifier
        dt_id_type = int

        fields_names = []
        if self._conf['model']['correctors_identifiers'] is not None:
            fields_names = fields_names + list(self._conf['model']['correctors_identifiers'])  # Correctors
        if self._conf['model']['predictor_identifier'] is not None:
            fields_names = fields_names + list(self._conf['model']['predictor_identifier'])        # Predictors
        if self._conf['model']['group_identifier'] is not None:
            fields_names = fields_names + list(self._conf['model']['group_identifier'])        # Predictors
        if self._conf['model']['dt_identifier'] is not None:
            fields_names = fields_names + list(self._conf['model']['dt_identifier'])        # Predictors

        # Load excel file
        if excel_file.endswith('.xls'):
            xls = ExcelSheet(excel_file)
        elif excel_file.endswith('.csv'):
            xls = CSVSheet(excel_file)
        else:
            raise ValueError('Specify one of the valid formats for excel file [xls,csv]')


        # Prepare fields type parameter
        if category_identifier:
            # If there is a category identifier, add the id identifier and the category identifier
            fields = {
                subject_identifier: subject_id_type,
                event_identifier: event_id_type,
                dt_identifier: dt_id_type,
                category_identifier: int
            }
        else:
            # Otherwise, just add the id identifier
            fields = {
                subject_identifier: subject_id_type,
                event_identifier: event_id_type,
                dt_identifier: dt_id_type,
            }
        for field in fields_names:
            fields[field] = float

        # Load the predictors and correctors for all subjects
        subjects = []
        end = end if end is None else end + 1
        for row in xls.get_rows(start=start, end=end, fieldstype=fields):
            # The subjects must have a non-empty ID
            if row[subject_identifier] != "" and row[event_identifier] != "":
                id_identifier = row[subject_identifier] + '_' + row[event_identifier]

                # Create path to nifti file
                nifti_path = join(data_folder, row[subject_identifier], study_prefix + 'step_' + event_identifier +
                                  '_dt_' + str(row[dt_identifier]) + extension)

                # Category
                category = row[category_identifier] if category_identifier else None

                # Create subject
                subj = Subject(id_identifier, nifti_path, extension, category=category)

                # Add prediction and correction parameters
                for param_name in fields_names:
                    subj.set_parameter(parameter_name=param_name, parameter_value=row[param_name])
                # Append subject to the subjects' list
                subjects.append(subj)

        # Cache subjects
        self._cached_subjects = subjects
        self._start = start
        self._end = end
        # Return the cached subjects
        return self._cached_subjects

    def get_corrector_random_effects(self, start=None, end=None, use_cache=True):
        """
        Returns the correctors random effects of the study for all the subjects between "start" and "end" [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            2D matrix with the correctors for all subjects
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        # correctors = [0]*len(subjects)
        if self._conf['model']['random_effects_identifiers'] is not None:
            correctors_names = self._conf['model']['random_effects_identifiers']
            correctors = list(map(lambda subject: subject.get_parameters(correctors_names), subjects))
        else:
            correctors = np.zeros((len(subjects),1))

        return np.asarray(correctors)

    def get_correctors_random_effects_names(self):
        """
        Returns the correctors' random effects names of this study

        Returns
        -------
        List<String>
            List of correctors' names
        """
        if self._conf['model']['correctors_identifiers'] is not None:
            return self._conf['model']['correctors_identifiers']
        else:
            return ['all-zeros']


    def get_groups(self, start=None, end=None, use_cache=True):
        """
        Returns the group of each observation with regards the random effects modeling

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            1D array with the group if for each observation
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        # correctors = [0]*len(subjects)
        if self._conf['model']['group_identifier'] is not None:
            group_names = self._conf['model']['group_identifier']
            groups = list(map(lambda subject: subject.get_parameters(group_names), subjects))
        else:
            groups = np.zeros((len(subjects),1))

        return np.asarray(groups)

    def get_subject_id(self, start=None, end=None, use_cache=True):
        """
        Returns the group of each observation with regards the random effects modeling

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            1D array with the group if for each observation
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        # correctors = [0]*len(subjects)
        if self._conf['model']['subject_identifier'] is not None:
            subject_names = self._conf['model']['subject_identifier']
            subjects_id = list(map(lambda subject: subject.get_parameters(subject_names), subjects))
        else:
            subjects_id = np.zeros((len(subjects),1))

        return np.asarray(subjects_id)

    def get_dt(self, start=None, end=None, use_cache=True):
        """
        Returns the group of each observation with regards the random effects modeling

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            1D array with the group if for each observation
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        # correctors = [0]*len(subjects)
        if self._conf['model']['group_identifier'] is not None:
            dt_names = self._conf['model']['dt_identifier']
            dts = list(map(lambda subject: subject.get_parameters(dt_names), subjects))
        else:
            dts = np.zeros((len(subjects),1))

        return np.asarray(dts)

