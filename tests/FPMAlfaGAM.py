import sys
sys.path.insert(1,'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
sys.stdout.flush()
from ExcelIO import ExcelSheet as Excel
from GAMProcessing import GAMProcessor as GAMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
import numpy as np


print 'Obtaining data from Excel file...'
from user_paths import DATA_DIR, EXCEL_FILE

filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[0][8:] : fn for fn in filenames}

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
print 'Initializing PolyGLM Processor...'
gamp = GAMP(subjects, predictors = [Subject.ADCSFIndex])

print 'Processing data...'
x1=85#103#
x2=x1+2
y1=101#45#
y2=y1+2
z1=45#81#
z2=z1+2
results = gamp.process(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)

print 'Saving results to files...'

affine = np.array(
		[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
		 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
		 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
		 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
)

from matplotlib import pyplot as plt
import numpy as np
corrected_data = gamp.corrected_values(results.correction_parameters, x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)
x=np.array(np.sort([sbj._attributes[sbj.ADCSFIndex.index] for sbj in subjects]))
plt.plot(x,np.squeeze(corrected_data),'k.')
plt.plot(x,gamp.__curve__(-1,x[:,np.newaxis],np.squeeze(results.regression_parameters)))
plt.show()

niiFile = nib.Nifti1Image

nib.save(niiFile(results.correction_parameters, affine), join('results', filename + '_cparams.nii'))
nib.save(niiFile(results.regression_parameters, affine), join('results', filename + '_rparams.nii'))
nib.save(niiFile(results.fitting_scores, affine), join('results', filename + '_fitscores.nii'))



filename = 'gam_splines_d5_s10'
with open(join('results', filename + '_userdefparams.txt'), 'wb') as f:
	f.write(str(gamp.user_defined_parameters) + '\n')

print 'Obtaining, filtering and saving z-scores and labels to display them...'
for fit_threshold in [0.99, 0.995, 0.999]:
	print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
	z_scores, labels = gamp.fit_score(results.fitting_scores, fit_threshold = fit_threshold, produce_labels = True)
	z_scores_2, labels_2 = gamp.fit_score(results.fitting_scores, fit_threshold = fit_threshold, produce_labels = True,cluster_threshold=0)

	print '    Saving z-scores and labels to file...'
	nib.save(niiFile(z_scores, affine), join('results', filename + '_zscores_' + str(fit_threshold) + '.nii'))
	nib.save(niiFile(labels, affine), join('results', filename + '_labels_' + str(fit_threshold) + '.nii'))
	nib.save(niiFile(z_scores, affine), join('results', filename + '_zscores_wo_cluster_' + str(fit_threshold) + '.nii'))
	nib.save(niiFile(labels, affine), join('results', filename + '_labels_wo_cluster' + str(fit_threshold) + '.nii'))


print 'Done.'

