#!/usr/bin/python
from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

from neat.Utils.niftiIO import get_atlas_image_labels
from neat.Utils.Clustering import RecursiveClustering, HierarchicalClustering
import numpy as np

from neat import helper_functions
import nibabel as nib

if __name__ == '__main__':

    HEMI_CHOICE = {
        'left': 'lh',
        'right': 'rh',
        '':''
    }

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Computes statistical maps for the fitting results '
                                                  ' computed by compute_fitting.py. By default '
                                                  ' uses all computed parameters inside the results'
                                                  ' folder specified in the configuration file.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file"
                                                             " used to load the data for this study.")

    arguments_parser.add_argument('--dirs', nargs='+', help='Specify one or several directories within the'
                                                            ' results directory specified in the '
                                                            ' configuration file from which the '
                                                            ' parameters should be loaded.')

    arguments_parser.add_argument('--map', help='Path relative to the output directory specified in the configuration '
                                                'file to the statistical map to be loaded')

    arguments_parser.add_argument('--hemi', default ='',choices=HEMI_CHOICE, help='Mandatory for surface-based analysis.'
                                                                                  'Please, speciy either left or right.')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    dirs = arguments.dirs
    map_name = arguments.map
    hemi = HEMI_CHOICE[arguments.hemi]


    """ LOAD DATA USING DATALOADER """
    subjects, covariate_names, covariates, processing_parameters, affine_matrix, output_dir, \
    results_io, type_data = helper_functions.load_data_from_config_file(config_file)
    template = helper_functions.load_template_from_config_file(config_file)


    """ LOAD STATISTICAL MAP """
    if map_name is not None:
        full_path = path.join(output_dir, map_name)
        if not path.isfile(full_path):
            print('{} does not exist or is not recognized as a file. Try again with a valid file.'.format(full_path))
            exit(1)

        map_data = results_io.loader(full_path).get_data()
        if type_data == 'vol':
            if map_data.shape[:3] != template.shape[:3]:
                print(
                    "The specified map and the template don't have the same dimensions. Try again with a valid statistical "
                    "map")
                exit(1)
        else:
            if map_data.shape[0] != template[0].shape[0]:
                print(
                    "The specified map and the template don't have the same dimensions. Try again with a valid statistical "
                    "map")
                exit(1)

        # Mask the map
        masked_map_data = 1 - np.ma.masked_values(map_data, 0.0).mask.astype(int)
        number_of_effective_curves = int(np.sum(masked_map_data))
        indices_effective_curves = np.where(masked_map_data==1)
        index_matrix = np.zeros((number_of_effective_curves,3), dtype=int)
        index_matrix[:,0] = indices_effective_curves[0].astype(int)
        index_matrix[:,1] = indices_effective_curves[1].astype(int)
        index_matrix[:,2] = indices_effective_curves[2].astype(int)

    else:
        number_of_effective_curves = int(np.prod(template.shape))
        indices_effective_curves = np.where(template>0)
        index_matrix = np.zeros((number_of_effective_curves,3))
        index_matrix[:, 0] = indices_effective_curves[0].astype(int)
        index_matrix[:, 1] = indices_effective_curves[1].astype(int)
        index_matrix[:, 2] = indices_effective_curves[2].astype(int)

    # Lists to store the necessary data to show the curves
    folder_name_list = []
    prediction_parameters = []
    correction_parameters = []
    processors = []

    """ LOAD CURVES """
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
            pathname[0], results_io, subjects, covariate_names, covariates,
            processing_parameters, type_data
        )
        folder_name_list.append(n)
        prediction_parameters.append(pred_p)
        correction_parameters.append(corr_p)
        processors.append(proc)


    # Initialize Clustering Algorithm
    n_clusters = 2
    n_clusters_prestep = 40
    cluster_technique = HierarchicalClustering(n_clusters, template=template)
    # cluster_technique = RecursiveClustering(n_clusters, template=template, n_clusters_hierarchical=n_clusters_prestep)

    # Get corrected grey matter data
    print('Loading curves...')
    tpoints = 100
    dict_curves = {}
    dict_curves_concatenated = {}
    for i in range(len(processors)):
        print(processors[i].get_name())
        dict_curves[processors[i].get_name()] = {}

        # Get curves
        axis, curve = processors[i].curve(
            prediction_parameters[i],
            tpoints=tpoints
        )
        if number_of_effective_curves != np.prod(template.shape):
            masked_curve = np.zeros((tpoints, number_of_effective_curves))
            for it_curve in range(number_of_effective_curves):
                masked_curve[:,it_curve] = curve[:,index_matrix[it_curve,0], index_matrix[it_curve,1], index_matrix[it_curve,2]]
        else:
            masked_curve = np.reshape(curve, (tpoints,-1))

        masked_curve = masked_curve.T
        dict_curves[processors[i].get_name()] = masked_curve

        #Algorithm
        print('Clustering...')
        results, image_clusters = cluster_technique.clusterize(masked_curve, index_matrix, x_axis = axis.T[:,0],
                                                               x_axis_name = processors[i].predictor_names[0])

        for figure_name, figure in image_clusters:
            figure.savefig(path.join(output_dir,folder_name_list[i], hemi + '-' + figure_name + '.png') if hemi != '' else path.join(
                output_dir, folder_name_list[i], figure_name + '.png'))

        for name, data in results:
            full_file_path = path.join(output_dir, folder_name_list[i], hemi + '-' + name + results_io.extension) if hemi != '' else path.join(
                output_dir, folder_name_list[i], name + results_io.extension)
            res_writer = results_io.writer(data, affine_matrix)
            res_writer.save(full_file_path)


