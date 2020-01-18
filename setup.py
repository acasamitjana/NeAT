from distutils.core import setup

setup(
    name='NeAT',
    version='0.0.0',
    packages=['neat', 'neat.Utils', 'neat.Fitters', 'neat.FitScores', 'neat.Processors', 'neat.Visualization',
              'neat.CrossValidation'],
    scripts=['neat-compare_statistical_maps.py', 'neat-compute_fitting.py', 'neat-compute_statistical_maps.py',
             'neat-generate_user_parameters.py', 'neat-search_hyperparameters.py', 'neat-show_curves.py',
             'neat-show_data_distribution.py', 'neat-show_visualizer.py'],
    url='',
    license='MIT',
    author='Image Processing Group',
    author_email='adria.casamitjana@upc.es',
    description=''
)
