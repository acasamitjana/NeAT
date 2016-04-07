import sys

sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
sys.path.insert(1, '/Users/acasamitjana/Repositories/neuroimatge/nonlinear2')
sys.stdout.flush()
from ExcelIO import ExcelSheet as Excel
from GAMProcessing import GAMProcessor as GAMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir
import nibabel as nib
import numpy as np

print 'Obtaining data from Excel file...'
from user_paths import DATA_DIR, EXCEL_FILE, CORRECTED_DIR

filenames = filter(isfile, map(lambda elem: join(CORRECTED_DIR, elem), listdir(CORRECTED_DIR)))
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
print 'Initializing GAM Processor...'

udp = (9, [2, 2, 0.5, 3])
gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)

print 'Processing data...'
x1 = 85  # 103#
x2 = x1 + 1
y1 = 101  # 45#
y2 = y1 + 1
z1 = 45  # 81#
z2 = z1 + 1
results = gamp.process(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)

print 'Saving results to files...'

affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

from matplotlib import pyplot as plt
import numpy as np

correction_parameters = np.zeros((results.correction_parameters.shape[0],200,200,200))
correction_parameters[:, x1:x2, y1:y2, z1:z2] = results.correction_parameters
corrected_data = gamp.corrected_values(correction_parameters,x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)
x = np.array(np.sort([sbj._attributes[sbj.ADCSFIndex.index] for sbj in subjects]))
plt.plot(x, np.squeeze(corrected_data), 'k.')
plt.plot(x, gamp.__curve__(-1, x[:, np.newaxis], np.squeeze(results.prediction_parameters)))
plt.show()

a=1