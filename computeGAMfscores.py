from nonlinear2.Utils.ExcelIO import ExcelSheet as Excel
from nonlinear2.FitScores.FitEvaluation import ftest
from nonlinear2.Processors.GAMProcessing import GAMProcessor as GAMP
from nonlinear2.Utils.Subject import Subject
from os.path import join, isfile, basename
from os import listdir
import nibabel as nib
import numpy as np
from scipy.stats import norm


print 'Obtaining data from Excel file...'

from user_paths import DATA_DIR, EXCEL_FILE, CORRECTED_DATA_DIR, RESULTS_DIR

niiFile = nib.Nifti1Image
affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

filenames = filter(isfile, map(lambda elem: join(CORRECTED_DATA_DIR, elem), listdir(CORRECTED_DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[1][:-4] : fn for fn in filenames}

exc = Excel(EXCEL_FILE)

subjects = []
for r in exc.get_rows(fieldstype={
    'id': (lambda s: str(s).strip().split('_')[0]),
    'diag': (lambda s: int(s) - 1),
    'age': int,
    'sex': (lambda s: 2 * int(s) - 1),
    'apoe4_bin': (lambda s: 2 * int(s) - 1),
    'escolaridad': int,
    'ad_csf_index_ttau': float
}):
    subjects.append(
        Subject(
            r['id'],
            filenames_by_id[r['id']],
            r.get('diag', None),
            r.get('age', None),
            r.get('sex', None),
            r.get('apoe4_bin', None),
            r.get('escolaridad', None),
            r.get('ad_csf_index_ttau', None)
        )
    )

print 'Loading precomputed parameters for GLM'
filename_prefix = 'PGAM\\gam_poly_d3_'

gam_correction_parameters = nib.load(join(RESULTS_DIR, filename_prefix + 'cparams.nii')).get_data()
gam_prediction_parameters = nib.load(join(RESULTS_DIR, filename_prefix + 'pparams.nii')).get_data()

with open(join(RESULTS_DIR,filename_prefix + 'userdefparams.txt'), 'rb') as f:
    user_defined_parameters = eval(f.read())

print 'Initializing GAM Processor...'
gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=user_defined_parameters)

print 'Computing F-scores'
fitting_scores = GAMP.evaluate_fit(
    evaluation_function=ftest,
    correction_processor=gamp,
    correction_parameters=gam_correction_parameters,
    prediction_processor=gamp,
    prediction_parameters=gam_prediction_parameters,
    # x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None,
    # origx = 0, origy = 0, origz = 0,
    gm_threshold=0.1,
    filter_nans=True,
    default_value=0.0,
    mem_usage = 128,
    # *args, **kwargs
)

print 'Saving inverted p-values to file'
nib.save(niiFile(fitting_scores, affine), join(RESULTS_DIR, filename_prefix + 'fitscores.nii'))

print 'Obtaining, filtering and saving z-scores and labels to display them...'
for fit_threshold in [0.99, 0.995, 0.999]:
    print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
    clusterized_fitting_scores, labels = GAMP.clusterize(
        fitting_scores=fitting_scores,
        default_value=0.0,
        fit_lower_threshold=fit_threshold,
        # fit_upper_threshold = None,
        cluster_threshold=100,
        produce_labels=True
    )
    # Compute Z-scores
    lim_value = norm.ppf(fit_threshold)
    valid_voxels = clusterized_fitting_scores != 0.0
    clusterized_fitting_scores[valid_voxels] = norm.ppf(clusterized_fitting_scores[valid_voxels]) - lim_value + 0.2

    print '    Saving z-scores and labels to file...'
    nib.save(niiFile(clusterized_fitting_scores, affine), join(RESULTS_DIR, filename_prefix + 'zscores_' + str(fit_threshold) + '.nii'))
    nib.save(niiFile(labels, affine), join(RESULTS_DIR, filename_prefix + 'labels_' + str(fit_threshold) + '.nii'))

print 'Done.'
