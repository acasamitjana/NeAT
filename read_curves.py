#!/usr/bin/python
from __future__ import print_function

from glob import glob
from os import path

from neat.Utils.niftiIO import get_atlas_image_labels
import numpy as np

from neat import helper_functions

if __name__ == '__main__':

    config_file = '/imatge/acasamitjana/Repositories/NeAT/config/adContinuum.yaml'
    base_dir = '/work/acasamitjana/NeAT/Tutorial/adContinuum'
    dirs = [path.join(base_dir, 'correction_PolyGLM-prediction_PolyGLM'),
            path.join(base_dir, 'correction_PolyGLM-prediction_GAM'),
            path.join(base_dir, 'correction_PolyGLM-prediction_PolySVR'),
            path.join(base_dir, 'correction_PolyGLM-prediction_GaussianSVR')
            ]




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

    index_RHipp = np.where(atlas == 38)
    x1_RHipp = np.min(index_RHipp[0])
    x2_RHipp = np.max(index_RHipp[0])
    y1_RHipp = np.min(index_RHipp[1])
    y2_RHipp = np.max(index_RHipp[1])
    z1_RHipp = np.min(index_RHipp[2])
    z2_RHipp = np.max(index_RHipp[2])


    index_RCaudate = np.where(atlas == 72)
    x1_RCaudate = np.min(index_RCaudate[0])
    x2_RCaudate = np.max(index_RCaudate[0])
    y1_RCaudate = np.min(index_RCaudate[1])
    y2_RCaudate = np.max(index_RCaudate[1])
    z1_RCaudate = np.min(index_RCaudate[2])
    z2_RCaudate = np.max(index_RCaudate[2])


    label_ROI_dict = {'RHipp': 38, 'RCaudate': 72}
    index_composite = {'RHipp': (x1_RHipp,y1_RHipp,z1_RHipp), 'RCaudate': (x1_RCaudate,y1_RCaudate,z1_RCaudate)}

    # Get corrected grey matter data
    print('Loading curves...')
    dict_curves = {}
    dict_curves_concatenated = {}
    for i in range(len(processors)):
        print(processors[i].get_name())
        dict_curves[processors[i].get_name()] = {}

        ############
        # Region 1 #
        ############
        # corrected_data = processors[i].corrected_values(
        #     correction_parameters[i],
        #     x1=x1_RHipp,
        #     x2=x2_RHipp,
        #     y1=y1_RHipp,
        #     y2=y2_RHipp,
        #     z1=z1_RHipp,
        #     z2=z2_RHipp
        # )

        # Get curves
        axis, curve = processors[i].curve(
            prediction_parameters[i],
            x1=x1_RHipp,
            x2=x2_RHipp,
            y1=y1_RHipp,
            y2=y2_RHipp,
            z1=z1_RHipp,
            z2=z2_RHipp,
            tpoints=100
        )
        dict_curves[processors[i].get_name()]['RHipp'] = curve
        curve_RHipp = curve.reshape((axis.shape[1], -1))


        ############
        # Region 1 #
        ############
        # corrected_data = processors[i].corrected_values(
        #     correction_parameters[i],
        #     x1=x1_RCaudate,
        #     x2=x2_RCaudate,
        #     y1=y1_RCaudate,
        #     y2=y2_RCaudate,
        #     z1=z1_RCaudate,
        #     z2=z2_RCaudate
        # )

        # Get curves
        axis, curve = processors[i].curve(
            prediction_parameters[i],
            x1=x1_RCaudate,
            x2=x2_RCaudate,
            y1=y1_RCaudate,
            y2=y2_RCaudate,
            z1=z1_RCaudate,
            z2=z2_RCaudate,
            tpoints=100
        )
        dict_curves[processors[i].get_name()]['RCaudate'] = curve
        curve_RCaudate = curve.reshape((axis.shape[1], -1))

        dict_curves_concatenated[processors[i].get_name()] = np.concatenate((curve_RHipp, curve_RCaudate), axis=1)




    import csv
    from os.path import join
    OUTPUT_DIR = '/work/acasamitjana/NeAT/'
    FILE = '_curves_RHipp_RCau.csv'
    fieldnames = ['X', 'Y', 'Z', 'ROI'] + list(range(100))

    for processor_name, ROI_dict in dict_curves.items():
        with open(join(OUTPUT_DIR, processor_name + FILE), 'w') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            for ROI_name, ROI_curves in ROI_dict.items():
                for x in range(ROI_curves.shape[1]):
                    for y in range(ROI_curves.shape[2]):
                        for z in range(ROI_curves.shape[3]):
                            if atlas[x+index_composite[ROI_name][0],y+index_composite[ROI_name][1], z+index_composite[ROI_name][2]] == label_ROI_dict[ROI_name]:
                                to_write = {}
                                to_write['X'] = x+index_composite[ROI_name][0]
                                to_write['Y'] = y+index_composite[ROI_name][1]
                                to_write['Z'] = z+index_composite[ROI_name][2]
                                to_write['ROI'] = ROI_name
                                for i in range(100):
                                    to_write[i] = ROI_curves[i,x,y,z]

                                csv_writer.writerow(to_write)


