import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh, det
from time import time

import hamiltonian as ham
import operators as op
import functions as func

from plots import *

begin = time()

calc_disconnected_EE_free()

end = time()
print "\nTotal time used:", end - begin
plt.show()
