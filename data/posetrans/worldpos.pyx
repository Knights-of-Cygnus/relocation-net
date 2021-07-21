cimport numpy as np
cimport cython
from libcpp cimport bool

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef assoc_z_pose(np.ndarray ndi, bool homo):
    ys = []
    for i, row in enumerate(ndi):
        xs = []
        for j, depth in enumerate(row):
            coord = [i, j, depth]
            if homo:
                coord.append(1)
            xs.append(coord)
        ys.append(xs)
    return ys
