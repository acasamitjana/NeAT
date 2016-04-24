from Utils.ExcelIO import ExcelSheet as Excel
from Processors.GAMProcessing import GAMProcessor as GAMP
from Utils.Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
from numpy import array as nparray


print 'Obtaining data from Excel file...'
# DATA_DIR = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'Nonlinear_NBA_15')
# EXCEL_FILE = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'work_DB_CSF.R1.final.xls')
# DATA_DIR = join('C:\\','Users','upcnet','FPM','data_backup','Non-linear', 'Nonlinear_NBA_15')
# EXCEL_FILE = join('C:\\','Users','upcnet','FPM','data_backup','Non-linear', 'work_DB_CSF.R1.final.xls')
DATA_DIR = join('/','Users','acasamitjana','FPM','Data_backup','Non-linear','Nonlinear_NBA_15')
CORRECTED_DIR = join('/','Users','acasamitjana','FPM','Data_backup','Non-linear','Nonlinear_NBA_15_corrected')
EXCEL_FILE = join('/','Users','acasamitjana','FPM','Data_backup','Non-linear','work_DB_CSF.R1.final.xls')

filenames = filter(isfile, map(lambda elem: join(CORRECTED_DIR, elem), listdir(CORRECTED_DIR)))
filenames_by_id = {basename(fn)[10:13] : fn for fn in filenames}

exc = Excel(EXCEL_FILE)

subjects = []
for r in exc.get_rows( fieldstype = {
				'id':(lambda s: str(s).strip().split('_')[0]),
				'diag':(lambda s: int(s) - 1),
				'age':int,
				'sex':(lambda s: 2*int(s) - 1),
				'apoe4_bin':(lambda s: 2*int(s) - 1),
				'escolaridad':int,
				'ad_csf_index_ttau':float
			 } ):
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

print 'Initializing GAM Processor...'
user_defined_parameters = [
	(9,[2,2,1,1]),
	(9,[2,2,1,2]),
	(9,[2,2,1,3]),
	(9,[2,2,1,4]),
	(9,[2,2,1,5]),

]

filenames = [
	'gam_splines_d1_s1',
	'gam_splines_d2_s1',
	'gam_splines_d3_s1',
	'gam_splines_d4_s1',
	'gam_splines_d5_s1',
]

for udp,filename in zip(user_defined_parameters,filenames):
	gamp = GAMP(subjects, regressors = [Subject.ADCSFIndex],user_defined_parameters=udp)

	print 'Processing data...'
	results = gamp.process()

	print 'Saving results to files...'

	affine = nparray(
			[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
			 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
			 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
			 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
	)

	niiFile = nib.Nifti1Image

	nib.save(niiFile(results.correction_parameters, affine), join('results', filename + '_cparams.nii'))
	nib.save(niiFile(results.regression_parameters, affine), join('results', filename + '_rparams.nii'))
	nib.save(niiFile(results.fitting_scores, affine), join('results', filename + '_fitscores.nii'))

	with open(join('results', filename + '_userdefparams.txt'), 'wb') as f:
		f.write(str(gamp.user_defined_parameters) + '\n')

	print 'Obtaining, filtering and saving z-scores and labels to display them...'
	for fit_threshold in [0.99, 0.995, 0.999]:
		print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
		z_scores, labels = gamp.fit_score(results.fitting_scores, fit_threshold = fit_threshold, produce_labels = True)

		print '    Saving z-scores and labels to file...'
		nib.save(niiFile(z_scores, affine), join('results', filename + '_zscores_' + str(fit_threshold) + '.nii'))
		nib.save(niiFile(labels, affine), join('results', filename + '__labels_' + str(fit_threshold) + '.nii'))


print 'Done.'
