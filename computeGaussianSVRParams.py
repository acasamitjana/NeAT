from os.path import join

import nibabel as nib
import Utils.DataLoader as DataLoader
from Processors.SVRProcessing import GaussianSVRProcessor as GSVRP
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

if __name__ == "__main__":

    # Define filename prefix
    filename_prefix = join(RESULTS_DIR, 'GSVR')

    print 'Obtaining data from Excel file...'
    subjects = DataLoader.getSubjects(corrected_data=True) # Used for prediction
    # subjects = DataLoader.getSubjects(corrected_data=False) # Used for correction and prediction

    print 'Initializing GaussianSVR Processor...'
    udp = udp = (2, 3, 3.16227766017, 0.0891666666667, 0.25) # Used for prediction
    # udp = (1, 0, 3.0, 0.08, 0.25) # Used for correction and prediction
    gsvrp = GSVRP(subjects,
                  predictors=[Subject.ADCSFIndex],
                  user_defined_parameters=udp)

    print 'Processing data...'
    results = gsvrp.process(n_jobs=3, mem_usage=512, cache_size=2096)
    C = gsvrp.user_defined_parameters[2]
    epsilon = gsvrp.user_defined_parameters[3]
    gamma = gsvrp.user_defined_parameters[4]

    print 'Saving results to files...'
    affine = DataLoader.getMNIAffine()
    niiFile = nib.Nifti1Image
    filename = 'gsvr_C' + str(C) + '_eps' + str(epsilon) + '_gamma' + str(gamma) + '_'
    nib.save(niiFile(results.correction_parameters, affine), join(filename_prefix, filename + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(filename_prefix, filename + 'pparams.nii'))

    with open(join(filename_prefix, filename + 'userdefparams.txt'), 'wb') as f:
        f.write(str(gsvrp.user_defined_parameters) + '\n')

    print 'Done.'
