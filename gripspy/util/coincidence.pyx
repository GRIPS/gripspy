cimport cython
from libc.stdlib cimport malloc, free


__all__ = ['find_nearest_in']


ctypedef fused orderable_numeric:
    cython.char
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble


@cython.boundscheck(False)
def find_nearest_in(orderable_numeric[:] a, orderable_numeric[:] b):
    """For two sorted arrays a and b, returns a list of the indices of the nearest element in b
    for each element in a.  The two arrays must be of the same numeric type.
    """
    cdef Py_ssize_t na = a.shape[0]
    cdef Py_ssize_t nb = b.shape[0]
    cdef Py_ssize_t *nearest = <Py_ssize_t *>malloc(na * sizeof(Py_ssize_t))

    cdef Py_ssize_t ia, ib

    for ia in range(na):
        for ib in range(nearest[ia - 1] if ia > 0 else 0, nb):
            if b[ib] >= a[ia] or ib == nb - 1:
                if ib > 0 and b[ib] - a[ia] > a[ia] - b[ib - 1]:
                    ib -= 1
                nearest[ia] = ib
                break

    try:
        return [nearest[i] for i in range(na)]
    finally:
        free(nearest)
