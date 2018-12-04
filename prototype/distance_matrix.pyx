cimport numpy as npc
import numpy as np
cimport cython
@cython.boundscheck(False) # turn off boundscheck for this function
@cython.wraparound(False)  # turn off wraparound for this function
@cython.nonecheck(False)   # turn off nonecheck for this function
def distance_matrix_45(npc.ndarray[npc.int64_t, ndim=2] npa, double tau=0.1):
    '''Distance square matrix calculating for custom distance metric
      - distance=0 for main diagonal (distance to self)
      - only upper part of matrix (acceptable for sklearn.cluster.DBSCAN)
      - euclidean distance only when DX and DY deviation is less than TAU (relatively)
      - otherwise distance not specified (for further scipy.sparse.coo_matrix usage)
    Input:
      npa:numpy.ndarray[ndim=2] - array of point coordinates (X, Y)
      tau:double - max relative deviation of DX and DY
    Output:
      numpy.ndarray[ndim=3] - array of (distance, idx_1, idx_2) in scipy.sparse.coo_matrix
    '''
    cdef long n = npa.shape[0]
    cdef long buf_size = n * 100
    cdef long[:, :] a = npa
    cdef npc.ndarray[npc.float64_t, ndim=2] result = np.empty((0, 3))
    cdef npc.ndarray[npc.float64_t, ndim=2] buf = np.empty((buf_size, 3))
    cdef long i, j, k = 0
    cdef long dx, dy
    # input size check
    if n > 25000: 
        return None
    # main diagonal
    for i in range(n):
        buf[i, 0] = 0.
        buf[i, 1] = i
        buf[i, 2] = i
    result = np.concatenate((result, buf[:n]))
    # rows and columns processing
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = abs(a[i, 0] - a[j, 0])
            dy = abs(a[i, 1] - a[j, 1])
            if (dx != 0 or dy != 0) and abs(dx-dy) < max(dx, dy)*tau:
                buf[k, 0] = (dx + dy)*0.707106781 
                buf[k, 1] = i
                buf[k, 2] = j
                k += 1
                if k == buf_size:
                    # buf copy
                    result = np.concatenate((result, buf))
                    k = 0
    # final buf copy
    if k > 0:
        result = np.concatenate((result, buf[:k]))
    return result