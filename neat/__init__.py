import numpy as np
import os

dirname = os.path.dirname(__file__)
print(os.path.join(dirname,'..','LICENSE.txt'))
with open(os.path.join(dirname,'..','LICENSE.txt')) as file:
    seed = file.read()
    np.random.seed(int(seed))
    if 7696726 != np.random.randint(0, 100000000):
        raise ValueError('LICENSE ERROR:'
                         'Please, visit https://imatge-upc.github.io/neat-tool/ to download the LICENSE file')
