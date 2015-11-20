print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

print 'Setting relevant parameters for the program...',

import database as db
from graphlib import NiftiGraph as NGraph
from subject import Subject
from numpy import ones
from os.path import join
from scipy.stats import ttest_ind, norm

# Set paths
WORK_DIR =  join('/', 'Users', 'Asier', 'Documents', 'TFG')
DATA_DIR =  join('Alan T', 'Nonlinear_NBA_15')
EXCEL_FILE = join('Alan T', 'work_DB_CSF.R1.final.xls')
OUTPUT_DIR = join('python', 'ttest')
OUTPUT_FILENAME = 'ttest_results_v3'

# Set region
x1, y1, z1 = [0]*3
x2, y2, z2 = [None]*3
#	rang = (30, 30, 30)
#	start = (37, 29, 62)
#	x2, y2, z2 = tuple(start[i] + rang[i] for i in range(3))
#	x1, y1, z1 = start


# Set thresholds for p-value and gray matter quantity (per unit volume)
pv_threshold = 0.01
gm_threshold = 0.2

# Set threshold for minimum number of nodes present in a cluster for this
# to be considered and shown in the results
num_nodes_cluster = 100

print 'Done.'

print 'Reading data and performing t-tests over it...'

in_data = db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2,
					 fields = {'id': (lambda s: str(s).strip().split('_')[0]),
					 		   'diag':(lambda s: int(s) - 1)})

lsd = len(Subject.diagnostics)
dims = in_data.dims[1:4]
out_data = [[ones(dims) for _ in range(i+1, lsd)] for i in range(lsd - 1)]

diag = [[] for _ in range(lsd)]
for i in range(len(in_data.subjects)):
	diag[in_data.subjects[i].diag].append(i)

prog_inc_x = 10000./dims[0]
prog_inc_y = prog_inc_x/dims[1]
prog_inc_z = prog_inc_y/dims[2]

for chunk in in_data.chunks():
	x, y, z = chunk.coords
	x -= x1
	y -= y1
	z -= z1

	dw, dx, dy, dz = chunk.data.shape
	valid_voxels = (sum(chunk.data) >= gm_threshold*dw).astype(int)
	non_valid_voxels = 1 - valid_voxels

	data = [[chunk.data[i] for i in l] for l in diag]
	for i in range(lsd - 1):
		for j in range(lsd - i - 1):
			tt_res = ttest_ind(data[i], data[i+j+1])
			out_data[i][j][x:(x+dx), y:(y+dy), z:(z+dz)] = tt_res.pvalue*valid_voxels + non_valid_voxels

	progress = prog_inc_x*(x+dx) + prog_inc_y*(y+dy) + prog_inc_z*(z+dz)
	print '    ' + str(int(progress)/100.) + '%'

print 'Done.'

del data
del x, y, z
del i, j
del tt_res

print 'Filtering for clusters of size >= ' + str(num_nodes_cluster) + '...',

lim_value = norm.ppf(1 - pv_threshold)

for i in range(len(out_data)):
	for j in range(len(out_data[i])):
		g = NGraph(out_data[i][j], pv_threshold)
		for scc in g.sccs():
			if len(scc) < num_nodes_cluster:
				for x, y, z in scc:
					out_data[i][j][x, y, z] = 0.0
			else:
				for x, y, z in scc:
					# z-score
					out_data[i][j][x, y, z] = norm.ppf(1 - out_data[i][j][x, y, z]) - lim_value + 0.2

print 'Done.'

print 'Storing results to file...',

for i in range(len(out_data)):
	for j in range(len(out_data[i])):
		fn = OUTPUT_FILENAME + '_' + Subject.diagnostics[i] + '_' + Subject.diagnostics[i+j+1] + '.nii'
		abs_fn = join(WORK_DIR, OUTPUT_DIR, fn)
		db.save_output_data(out_data[i][j], abs_fn)

print 'Done.'

print
print 'Program finished without errors. Thank you for your patience!'
