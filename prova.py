import numpy as np
import os

from os.path import join
from nibabel.freesurfer import io, mghformat


DIR = '/projects/neuro/AlfaStudy/aging_apoe_data_surface/data'
OUT_DIR = '/projects/neuro/AlfaStudy/aging_apoe_data_surface/data_mean_centered'

fs_files = os.listdir(DIR)
dataset_array = np.zeros((len(fs_files),163842))

for it_file, file in enumerate(fs_files):
    dataset_array[it_file] = io.read_morph_data(join(DIR,file))
    print(it_file)

dataset_array = dataset_array - np.mean(dataset_array,axis=0,keepdims=True)
for it_file, file in enumerate(fs_files):
    io.write_morph_data(join(OUT_DIR,file), dataset_array[it_file])
    print(it_file)
