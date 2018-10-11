#!/usr/bin/python
from __future__ import print_function

import os
import os.path as path
from argparse import ArgumentParser

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from neat import helper_functions
from neat.Utils.niftiIO import get_atlas_image_labels
from neat.Utils.Subject import Chunks



def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return list(map((lambda x : x * cofficient), v))

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            #print "i =", i, ", projection vector =", proj_vec
            temp_vec = list(map(lambda x, y : x - y, temp_vec, proj_vec))
            #print "i =", i, ", temporary vector =", temp_vec
        Y.append(temp_vec)
    return Y


if __name__ == '__main__':

    HEMI_CHOICE = {
        'left': 'lh',
        'right': 'rh',
        '': ''
    }

    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Computes the fitting parameters for the data '
                                            'provided in the configuration file. These fitting parameters'
                                            ' can be computed for all subjects in the study (default behaviour)'
                                            ' or you can specify for which categories should the parameters be'
                                            ' computed ')
    arg_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                       ' used to load the data for this study.')

    arg_parser.add_argument('--categories', nargs='+', type=int, help='Category or categories (as they are represented '
                                                                      'in the Excel file) for which the fitting '
                                                                      'parameters should be computed')
    arg_parser.add_argument('--parameters', help='Path to the txt file within the results directory '
                                                 'that contains the user defined '
                                                 'parameters to load a pre-configured '
                                                 'correction and prediction processor')
    arg_parser.add_argument('--prefix', default='', help='Prefix used in the result files')

    arg_parser.add_argument('--hemi', default='', choices=HEMI_CHOICE, help='Mandatory for surface-based analysis.')


    arguments = arg_parser.parse_args()
    config_file = arguments.configuration_file
    categories = arguments.categories
    parameters = arguments.parameters
    prefix = arguments.prefix
    hemi = HEMI_CHOICE[arguments.hemi]

    """ LOAD DATA USING DATALOADER """
    subjects, covariate_names, covariates, processing_parameters,  affine_matrix, output_dir, \
    results_io, type_data = helper_functions.load_data_from_config_file(config_file)

    atlas, aal_labels_dict = get_atlas_image_labels(results_io,
                                                    '/imatge/acasamitjana/work/NeAT/Atlases/aal.nii',
                                                    '/imatge/acasamitjana/work/NeAT/Atlases/aal.csv',
                                                    )


    covariates = np.concatenate((covariates,np.ones((covariates.shape[0],1))), axis=1)
    covariates = np.asarray(gs(covariates.T)).T

    single_voxel = False
    if single_voxel:
        image_center_coordinates = np.asarray([
            int(np.floor(1.0 * atlas.shape[0]/2))+1,
            int(np.floor(1.0 * atlas.shape[1] / 2)) + 1,
            int(np.floor(1.0 * atlas.shape[2] / 2)) + 1,
        ])
        mm_coordinates = np.asarray([21,-36,6,1])
        voxel_coordinates = list(map(int, np.round(np.linalg.inv(affine_matrix).dot(mm_coordinates))))

        print(voxel_coordinates)
        x1 = voxel_coordinates[0]
        x2 = x1 + 1
        y1 = voxel_coordinates[1]
        y2 = y1 + 1
        z1 = voxel_coordinates[2]
        z2 = z1 + 1

        ROIlabel = atlas[x1, y1, z1]
        ROI = aal_labels_dict[ROIlabel]
        index_flattened_ROI = np.asarray([1])

    else:
        index_ROI = np.where(atlas == 38)

        x1 = np.min(index_ROI[0])
        x2 = np.max(index_ROI[0])
        y1 = np.min(index_ROI[1])
        y2 = np.max(index_ROI[1])
        z1 = np.min(index_ROI[2])
        z2 = np.max(index_ROI[2])

        index_flattened_ROI = np.zeros((x2-x1+1,y2-y1+1,z2-z1+1))
        for it_i in zip(index_ROI[0], index_ROI[1], index_ROI[2]):
            index_flattened_ROI[it_i[0]-x1, it_i[1]-y1, it_i[2]-z1] = 1

        index_flattened_ROI = np.reshape(index_flattened_ROI,(np.prod(index_flattened_ROI.shape),-1))

    chunks = Chunks(
        subjects,
        x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2
    )

    chunk = next(chunks)
    data = chunk.data
    dims = data.shape
    obs = np.reshape(data,(dims[0],-1))

    indices_apoe0 = covariates[:,covariate_names.index('0 Apoe-4')]
    indices_apoe1 = covariates[:,covariate_names.index('1 Apoe-4')]
    indices_apoe2 = covariates[:,covariate_names.index('2 Apoe-4')]


    from sklearn.linear_model import LinearRegression

    N_obs = obs.shape[1]
    print(N_obs)
    t_value_list1 = np.zeros((N_obs))
    t_value_list2 = np.zeros((N_obs))
    beta_list = np.zeros((N_obs,len(covariate_names)+1))

    for it_obs in range(N_obs):
        if np.mod(it_obs, 1000) == 0:
            print(str(it_obs) + '/' + str(N_obs))

        if index_flattened_ROI[it_obs] == 1:

            lr = LinearRegression(fit_intercept=False)
            lr.fit(covariates, obs[:,it_obs])
            error = obs[:,it_obs] - lr.predict(covariates)
            s = np.std(error)
            beta = lr.coef_
            contrast1 = np.asarray([-1,0,1]+[0]*(len(covariate_names)-3+1))
            contrast2 = np.asarray([1,0,-1]+[0]*(len(covariate_names)-3+1))


            t_value1 = np.dot(contrast1,beta) / (s*np.sqrt(np.dot(contrast1.T,np.dot(np.linalg.pinv(np.dot(covariates.T,covariates)),contrast1))))
            t_value2 = np.dot(contrast2,beta) / (s*np.sqrt(np.dot(contrast2.T,np.dot(np.linalg.pinv(np.dot(covariates.T,covariates)),contrast2))))
            t_value_list1[it_obs] = t_value1
            t_value_list2[it_obs] = t_value2
            beta_list[it_obs] = beta


    t_value_saved1 = np.zeros_like(atlas)
    t_value_saved2 = np.zeros_like(atlas)

    it_i = 0
    for it_obs in range(N_obs):
        if index_flattened_ROI[it_obs] == 1:
            t_value_saved1[index_ROI[0][it_i], index_ROI[1][it_i], index_ROI[2][it_i]] = t_value_list1[it_obs]
            t_value_saved2[index_ROI[0][it_i], index_ROI[1][it_i], index_ROI[2][it_i]] = t_value_list2[it_obs]
            it_i += 1

    img = nib.Nifti1Image(t_value_saved1,affine_matrix)
    nib.save(img, '/work/acasamitjana/NeAT/Refactoring/prova2.nii.gz')
    img = nib.Nifti1Image(t_value_saved2,affine_matrix)
    nib.save(img, '/work/acasamitjana/NeAT/Refactoring/prova1.nii.gz')


    print()
    print(max(t_value_list1))
    print(max(t_value_list2))

    t_argmax = np.argsort(t_value_list2)[0]
    print(t_value_list1[t_argmax])
    print(beta_list[t_argmax])





    plt.figure()
    plt.plot(0*np.ones(int(np.sum(indices_apoe0))), obs[np.where(indices_apoe0 == 1)[0],t_argmax] - np.dot(covariates[np.where(indices_apoe0 == 1)[0]],np.dot(contrast1,beta_list[t_argmax])),'r')
    plt.plot(1*np.ones(int(np.sum(indices_apoe1))), obs[np.where(indices_apoe1 == 1)[0],t_argmax] - np.dot(covariates[np.where(indices_apoe1 == 1)[0]],np.dot(contrast1, beta_list[t_argmax])),'b')
    plt.plot(2*np.ones(int(np.sum(indices_apoe2))), obs[np.where(indices_apoe2 == 1)[0],t_argmax] - np.dot(covariates[np.where(indices_apoe2 == 1)[0]],np.dot(contrast1, beta_list[t_argmax])),'g')
    plt.show()






