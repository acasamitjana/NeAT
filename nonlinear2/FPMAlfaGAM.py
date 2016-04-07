from ExcelIO import ExcelSheet as Excel
from FitEvaluation import ftest
from GAMProcessing import GAMProcessor as GAMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir
import nibabel as nib
import numpy as np

print 'Obtaining data from Excel file...'
from user_paths import DATA_DIR, EXCEL_FILE, CORRECTED_DIR

niiFile = nib.Nifti1Image
affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

filenames = filter(isfile, map(lambda elem: join(CORRECTED_DIR, elem), listdir(CORRECTED_DIR)))
filenames_by_id = {basename(fn)[10:13]: fn for fn in filenames}

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
print 'Initializing GAM Splines Processor...'
user_defined_parameters = [(9, [2, 2, 10, 1]),
                           (9, [2, 2, 10, 2]),
                           (9, [2, 2, 10, 3]),
                           (9, [2, 2, 10, 4]),
                           (9, [2, 2, 10, 5])
                           ]
filenames = ['gam_splines_d1_s10',
             'gam_splines_d2_s10',
             'gam_splines_d3_s10',
             'gam_splines_d4_s10',
             'gam_splines_d5_s10',
             ]

for udf, filename in zip(user_defined_parameters, filenames):
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udf)
    print 'Processing data...'
    results = gamp.process()







print 'Initializing GAM Polynomial Processor...'
user_defined_parameters = [
    (9, [1, 1, 3]),

]

filenames = [
    'gam_poly_d3',
]

for udp, filename in zip(user_defined_parameters, filenames):
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)
    results = gamp.process()

