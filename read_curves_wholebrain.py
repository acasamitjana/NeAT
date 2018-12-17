#!/usr/bin/python
from __future__ import print_function

from glob import glob
from os import path

from neat.Utils.niftiIO import get_atlas_image_labels
import numpy as np

from neat import helper_functions
import nibabel as nib

if __name__ == '__main__':

    config_file = '/imatge/acasamitjana/Repositories/NeAT/config/adContinuum.yaml'
    base_dir = '/work/acasamitjana/NeAT/Tutorial/adContinuum'
    dirs = [#path.join(base_dir, 'correction_PolyGLM-prediction_PolyGLM'),
            path.join(base_dir, 'correction_PolyGLM-prediction_GAM'),
            path.join(base_dir, 'correction_PolyGLM-prediction_PolySVR'),
            path.join(base_dir, 'correction_PolyGLM-prediction_GaussianSVR')
            ]

    dict_dir = {path.basename(d):d for d in dirs}




    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir, results_io, type_data = helper_functions.load_data_from_config_file(config_file)


    # Lists to store the necessary data to show the curves
    names = []
    prediction_parameters = []
    correction_parameters = []
    processors = []

    """ LOAD DATA TO SHOW CURVES """
    print('Loading results data...')
    print()
    for directory in dirs:
        full_path = path.join(output_dir, directory)
        pathname = glob(path.join(full_path, '*prediction_parameters.mha'))
        # If there is no coincidence, ignore this directory
        if len(pathname) == 0:
            print('{} does not exist or contain any result.'.format(full_path))
            continue
        n, _, pred_p, corr_p, proc = helper_functions.get_results_from_path(
            pathname[0], results_io, subjects, predictors_names, correctors_names, predictors, correctors,
            processing_parameters, type_data
        )
        names.append(n)
        prediction_parameters.append(pred_p)
        correction_parameters.append(corr_p)
        processors.append(proc)


    #Get curves from RightHipoocampus and Caudate
    atlas, aal_labels_dict = get_atlas_image_labels(results_io,
                                                    '/imatge/acasamitjana/work/NeAT/Atlases/aal.nii',
                                                    '/imatge/acasamitjana/work/NeAT/Atlases/aal.csv',
                                                    )


    # Get corrected grey matter data
    print('Loading curves...')
    dict_curves = {}
    dict_curves_concatenated = {}
    for i in range(len(processors)):
        print(processors[i].get_name())
        dict_curves[processors[i].get_name()] = {}

        # Get curves
        axis, curve = processors[i].curve(
            prediction_parameters[i],
            tpoints=100
        )
        dict_curves[processors[i].get_name()] = curve


    import csv
    from os.path import join
    OUTPUT_DIR = '/work/acasamitjana/NeAT/'
    FILE = '_curves_whole_brain.csv'
    fieldnames = ['X', 'Y', 'Z', 'ROI'] + list(range(100))

    for processor_name, brain_curves in dict_curves.items():
        stats_map = np.asarray(nib.load(join(dict_dir[processor_name],'z-scores_0.999.nii.gz')).dataobj)

        with open(join(OUTPUT_DIR, processor_name + FILE), 'w') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            for x in range(stats_map.shape[0]):
                for y in range(stats_map.shape[1]):
                    for z in range(stats_map.shape[2]):
                        if stats_map[x,y,z] != 0 and atlas[x,y,z] != 0:
                            to_write = {}
                            to_write['X'] = x
                            to_write['Y'] = y
                            to_write['Z'] = z
                            to_write['ROI'] = aal_labels_dict[atlas[x,y,z]]
                            for i in range(100):
                                to_write[i] = brain_curves[i,x,y,z]

                            csv_writer.writerow(to_write)


