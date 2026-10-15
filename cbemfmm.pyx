import numpy as np
from scipy.sparse import csr_matrix

cimport numpy as np

np.import_array()

from libc.stddef cimport size_t

# rules 3, 6 and 9 put points on the facet edges, where the solid angle
# subtended by an edge neighbor is undefined
GAUSS_EN = (1, 4, 7, 13, 25, 48)
GAUSS_PN = (1, 3, 4, 6, 7, 9, 13, 25, 48)


cdef inline double *_dptr(np.ndarray a):
    return <double*>np.PyArray_DATA(a)


cdef inline size_t *_iptr(np.ndarray a):
    return <size_t*>np.PyArray_DATA(a)


def _check_gauss(gauss, allowed):
    if gauss not in allowed:
        raise ValueError(f"gauss must be one of {allowed}, got {gauss}")


def _vectors(name, a, rows=None):
    # rows x 3 float64 in C order, rows=None for any number
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or rows not in (None, a.shape[0]):
        raise ValueError(f"{name} must be {rows or 'N'} x 3, got shape {a.shape}")
    return a


def _vector(name, a):
    a = np.ascontiguousarray(a, dtype=np.float64).ravel()
    if a.size != 3:
        raise ValueError(f"{name} must hold 3 values, got {a.size}")
    return a


def _indices(name, a, rows, upper):
    # rows x K indices below upper, as uintp in C order
    a = np.asarray(a)
    if a.dtype.kind not in "iu":
        raise ValueError(f"{name} must hold integers, got {a.dtype}")
    if a.ndim != 2 or a.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got shape {a.shape}")
    if a.size and (a.min() < 0 or a.max() >= upper):
        raise ValueError(f"{name} must index 0 to {upper - 1}")
    a = np.ascontiguousarray(a)
    # the indices are non-negative, so int64 can be read as uintp in place
    if a.dtype.itemsize == np.dtype(np.uintp).itemsize:
        return a.view(np.uintp)
    return a.astype(np.uintp)


def _mesh(P, t, normal, center, neighbor, area):
    P = _vectors("P", P)
    t = _indices("t", t, np.shape(t)[0], P.shape[0])
    if t.shape[1] != 3:
        raise ValueError(f"t must be T x 3, got shape {t.shape}")
    T = t.shape[0]
    normal = _vectors("normal", normal, T)
    center = _vectors("center", center, T)
    neighbor = _indices("neighbor", neighbor, T, T)
    area = np.ascontiguousarray(area, dtype=np.float64).ravel()
    if area.size != T:
        raise ValueError(f"area must hold {T} values, got {area.size}")
    return P, t, normal, center, neighbor, area


def neighbor_ints_En (P, t, normal, center, neighbor, area, int gauss) -> csr_matrix:
    """
        MESHNEIGHBORINTS_EN Wrapper for C compiled neighborints_En
        P          - N x 3 points
        t          - T x 3 triangles
        normals    - T x 3 triangle normals
        Center     - T x 3 triangle centers
        ineighborE - T x M neighbor indices for each triangle
        Area       - T x 1 triangle areas
        gauss      - 1 x 1 number of gaussian cubature points
        options supported for gauss: 1,4,7,13,25,48
        Inputs of any float or integer type and order are converted;
        C-ordered float64 and 64-bit integer arrays are used as they are.
          GNP 2026
        SP 2026
    """
    _check_gauss(gauss, GAUSS_EN)
    P, t, normal, center, neighbor, area = _mesh(P, t, normal, center, neighbor, area)

    cdef size_t T = t.shape[0]
    cdef size_t M = neighbor.shape[1] # RnumberE

    IE = np.zeros((T, M), dtype=np.float64)
    IC = np.zeros((T, M), dtype=np.float64)

    cdef double *pP = _dptr(P)
    cdef size_t *pt = _iptr(t)
    cdef double *pnormal = _dptr(normal)
    cdef double *pcenter = _dptr(center)
    cdef size_t *pneighbor = _iptr(neighbor)
    cdef double *parea = _dptr(area)
    cdef double *pIE = _dptr(IE)
    cdef double *pIC = _dptr(IC)
    with nogil:
        cneighbor_ints_En(pP, pt, pnormal, pcenter, pneighbor, parea, T, M,
                          gauss, pIE, pIC)

    const = 1 / (4*np.pi);
    IE    = IE * area[:, None] / area[neighbor]

    ii    = neighbor.ravel()
    jj    = np.repeat(np.arange(T, dtype=np.uintp), M)

    data = const * (-IC.ravel() + IE.ravel())
    EC    = csr_matrix((data, (ii, jj)), shape=(T, T))

    return EC

def neighbor_ints_Pn (P, t, normal, center, neighbor, area, int gauss) -> csr_matrix:
    """
        MESHNEIGHBORINTS_PN Wrapper for C compiled neighborints_Pn
        P          - N x 3 points
        t          - T x 3 triangles
        normals    - T x 3 triangle normals
        Center     - T x 3 triangle centers
        ineighborP - T x M neighbor indices for each triangle
        Area       - T x 1 triangle areas
        gauss      - 1 x 1 number of gaussian cubature points
        options supported for gauss: 1,3,4,6,7,9,13,25,48
        Inputs of any float or integer type and order are converted;
        C-ordered float64 and 64-bit integer arrays are used as they are.

        Row n of ineighborP lists the observation facets nb near source
        facet n, as in the MATLAB meshneighborints_Pn. The result PC
        (T x T) corrects the plain potential at every facet:

            PC[nb, n] = 1/(4pi) * (<Is_n(1/r)>_nb - A_n / |C_nb - C_n|)

        where <.>_nb is the gauss average over observation facet nb. The
        center-point term is zero for nb == n.
          GNP 2026
        SP 2026
    """
    _check_gauss(gauss, GAUSS_PN)
    P, t, normal, center, neighbor, area = _mesh(P, t, normal, center, neighbor, area)

    cdef size_t T = t.shape[0]
    cdef size_t M = neighbor.shape[1] # RnumberP

    IP = np.zeros((T, M), dtype=np.float64)
    IPC = np.zeros((T, M), dtype=np.float64)

    cdef double *pP = _dptr(P)
    cdef size_t *pt = _iptr(t)
    cdef double *pnormal = _dptr(normal)
    cdef double *pcenter = _dptr(center)
    cdef size_t *pneighbor = _iptr(neighbor)
    cdef double *parea = _dptr(area)
    cdef double *pIP = _dptr(IP)
    cdef double *pIPC = _dptr(IPC)
    with nogil:
        cneighbor_ints_Pn(pP, pt, pnormal, pcenter, pneighbor, parea, T, M,
                          gauss, pIP, pIPC)

    const = 1 / (4*np.pi);

    ii    = neighbor.ravel()
    jj    = np.repeat(np.arange(T, dtype=np.uintp), M)

    data = const * (-IPC.ravel() + IP.ravel())
    PC    = csr_matrix((data, (ii, jj)), shape=(T, T))

    return PC

def potint(r1, r2, r3, normal, obs):
    """
        N
        r1 - 1 x 3 triangle vertex
        r2 - 1 x 3 triangle vertex
        r3 - 1 x 3 triangle vertex
        normal - 1 x 3 triangle normal, cross(r2 - r1, r3 - r1) normalized
        obs - N x 3 matrix of observation points
        -------------------------------------------------------------
        I - N x 1 vector of neighbor integrals I = Is(1/r)
        Irho - N x 3 matrix of neighbor integrals Irho = Is((y - p)/r),
               p the projection of the observation point on the plane

        SP 2026
    """
    r1, r2, r3 = _vector("r1", r1), _vector("r2", r2), _vector("r3", r3)
    normal = _vector("normal", normal)
    obs = _vectors("obs", np.atleast_2d(obs))

    cdef size_t N = obs.shape[0]
    I = np.zeros((N, 1), dtype=np.float64)
    Irho = np.zeros((N, 3), dtype=np.float64)

    cdef double *pr1 = _dptr(r1)
    cdef double *pr2 = _dptr(r2)
    cdef double *pr3 = _dptr(r3)
    cdef double *pnormal = _dptr(normal)
    cdef double *pobs = _dptr(obs)
    cdef double *pI = _dptr(I)
    cdef double *pIrho = _dptr(Irho)
    with nogil:
        c_potint(N, pr1, pr2, pr3, pnormal, pobs, pI, pIrho)

    return I, Irho


def potint2(r1, r2, r3, normal, obs):
    """
        N
        r1 - 1 x 3 triangle vertex
        r2 - 1 x 3 triangle vertex
        r3 - 1 x 3 triangle vertex
        normal - 1 x 3 triangle normal, cross(r2 - r1, r3 - r1) normalized
        obs - N x 3 matrix of observation points
        -------------------------------------------------------------
        Int - N x 3 matrix of gradient integrals Int = grad(Is(1/r))
              For points in the plane of the triangle the normal part is
              the principal value 0.

        SP 2026
    """
    r1, r2, r3 = _vector("r1", r1), _vector("r2", r2), _vector("r3", r3)
    normal = _vector("normal", normal)
    obs = _vectors("obs", np.atleast_2d(obs))

    cdef size_t N = obs.shape[0]
    Int = np.zeros((N, 3), dtype=np.float64)

    cdef double *pr1 = _dptr(r1)
    cdef double *pr2 = _dptr(r2)
    cdef double *pr3 = _dptr(r3)
    cdef double *pnormal = _dptr(normal)
    cdef double *pobs = _dptr(obs)
    cdef double *pInt = _dptr(Int)
    with nogil:
        c_potint2(N, pr1, pr2, pr3, pnormal, pobs, pInt)
    return Int
