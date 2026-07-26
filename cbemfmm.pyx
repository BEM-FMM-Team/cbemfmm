import numpy as np
from scipy.sparse import csr_matrix

cimport numpy as np

np.import_array()

from libc.stddef cimport size_t

from .cbemfmm cimport cneighbor_ints_En, c_potint, c_potint2


def neighbor_ints_En (
        np.ndarray[np.float64_t, ndim=2] P,
        np.ndarray[np.uintp_t, ndim=2] t,
        np.ndarray[np.float64_t, ndim=2] normal,
        np.ndarray[np.float64_t, ndim=2] center,
        np.ndarray[np.uintp_t, ndim=2] neighbor,
        np.ndarray[np.float64_t, ndim=1] area,
        int gauss
        ) -> csr_matrix:
    """
        MESHNEIGHBORINTS_EN Wrapper for C compiled neighborints_En
        P          - N x 3 points
        t          - T x 3 triangles
        normals    - T x 3 triangle normals
        Center     - T x 3 triangle centers
        ineighborE - T x M neighbor indices for each triangle
        Area       - T x 1 triangle areas
        gauss      - 1 x 1 number of gaussian cubature points
        options supported for gauss: 1,3,4,6,7,9,13,25,48
          GNP 2026
        SP 2026
    """
    if gauss not in [1,3,4,6,7,9,13,25,48]:
            raise RuntimeError("gauss must be one of: 1,3,4,6,7,9,13,25,48")

    P = np.asfortranarray(P, dtype=np.float64)
    t = np.asfortranarray(t, dtype=np.uintp)
    normal = np.asfortranarray(normal, dtype=np.float64)
    center = np.asfortranarray(center, dtype=np.float64)
    neighbor = np.asfortranarray(neighbor, dtype=np.uintp)
    area = np.asarray(area, dtype=np.float64)

    N = P.shape[0]
    T = t.shape[0]
    M = neighbor.shape[1] # RnumberE

    IE = np.zeros((T, M), dtype=np.float64, order="F")
    IC = np.zeros((T, M), dtype=np.float64, order="F")

    cneighbor_ints_En (
            <double*>np.PyArray_DATA(P),
            <size_t*>np.PyArray_DATA(t),
            <double*>np.PyArray_DATA(normal),
            <double*>np.PyArray_DATA(center),
            <size_t*>np.PyArray_DATA(neighbor),
            <double*>np.PyArray_DATA(area),
            N,
            T,
            M,
            gauss,
            <double*>np.PyArray_DATA(IE),
            <double*>np.PyArray_DATA(IC)
            )

    area_neighbor = area[neighbor]
    area_self     = np.repeat(area[:, None], M, axis=1)   # (T, M)

    const = 1 / (4*np.pi);
    IE    = IE * area_self / area_neighbor

    ii    = neighbor.T.flatten(order="F")
    jj    = np.repeat(np.arange(t.shape[0], dtype=np.uintp), M)

    data = const * (-IC.T.flatten(order="F") + IE.T.flatten(order="F"))
    EC    = csr_matrix((data, (ii, jj)), shape=(T, T))

    return EC

def neighbor_ints_Pn (
        np.ndarray[np.float64_t, ndim=2] P,
        np.ndarray[np.uintp_t, ndim=2] t,
        np.ndarray[np.float64_t, ndim=2] normal,
        np.ndarray[np.float64_t, ndim=2] center,
        np.ndarray[np.uintp_t, ndim=2] neighbor,
        np.ndarray[np.float64_t, ndim=1] area,
        int gauss
        ) -> csr_matrix:
    """
        MESHNEIGHBORINTS_PN Wrapper for C compiled neighborints_En
        P          - N x 3 points
        t          - T x 3 triangles
        normals    - T x 3 triangle normals
        Center     - T x 3 triangle centers
        ineighborE - T x M neighbor indices for each triangle
        Area       - T x 1 triangle areas
        gauss      - 1 x 1 number of gaussian cubature points
        options supported for gauss: 1,3,4,6,7,9,13,25,48
          GNP 2026
        SP 2026
    """
    if gauss not in [1,3,4,6,7,9,13,25,48]:
            raise RuntimeError("gauss must be one of: 1,3,4,6,7,9,13,25,48")

    P = np.asfortranarray(P, dtype=np.float64)
    t = np.asfortranarray(t, dtype=np.uintp)
    normal = np.asfortranarray(normal, dtype=np.float64)
    center = np.asfortranarray(center, dtype=np.float64)
    neighbor = np.asfortranarray(neighbor, dtype=np.uintp)
    area = np.asarray(area, dtype=np.float64)

    N = P.shape[0]
    T = t.shape[0]
    M = neighbor.shape[1] # RnumberE

    IP = np.zeros((T, M), dtype=np.float64, order="F")
    IPC = np.zeros((T, M), dtype=np.float64, order="F")

    cneighbor_ints_En (
            <double*>np.PyArray_DATA(P),
            <size_t*>np.PyArray_DATA(t),
            <double*>np.PyArray_DATA(normal),
            <double*>np.PyArray_DATA(center),
            <size_t*>np.PyArray_DATA(neighbor),
            <double*>np.PyArray_DATA(area),
            N,
            T,
            M,
            gauss,
            <double*>np.PyArray_DATA(IP),
            <double*>np.PyArray_DATA(IPC)
            )

    area_neighbor = area[neighbor]
    area_self     = np.repeat(area[:, None], M, axis=1)   # (T, M)

    const = 1 / (4*np.pi);
    IP    = IP * area_self / area_neighbor

    ii    = neighbor.T.flatten(order="F")
    jj    = np.repeat(np.arange(t.shape[0], dtype=np.uintp), M)

    data = const * (-IPC.T.flatten(order="F") + IP.T.flatten(order="F"))
    PC    = csr_matrix((data, (ii, jj)), shape=(T, T))

    return PC

def potint(
        np.ndarray[np.float64_t, ndim=2] r1,
        np.ndarray[np.float64_t, ndim=2] r2,
        np.ndarray[np.float64_t, ndim=2] r3,
        np.ndarray[np.float64_t, ndim=2] normal,
        np.ndarray[np.float64_t, ndim=2] obs,
        ):
    """
        N
        r1 - 1 x 3 triangle vertex
        r2 - 1 x 3 triangle vertex
        r3 - 1 x 3 triangle vertex
        normal - 1 x 3 triangle normal
        obs - N x 3 matrix of observation points
        -------------------------------------------------------------
        I - N x 1 vector of neighbor integrals I = Is(1/r)
        Irho - N x 3 matrix of neighbor integrals I = Is(vec(r)/r)

        SP 2026
    """
    N = obs.shape[0]

    r1 = np.asfortranarray(r1, dtype=np.float64)
    r2 = np.asfortranarray(r2, dtype=np.float64)
    r3 = np.asfortranarray(r3, dtype=np.float64)
    normal = np.asfortranarray(normal, dtype=np.float64)
    obs = np.asfortranarray(obs, dtype=np.float64)

    I = np.zeros((N, 1), dtype=np.float64, order="F")
    Irho = np.zeros((N, 3), dtype=np.float64, order="F")

    c_potint(
        N,
        <double*>np.PyArray_DATA(r1),
        <double*>np.PyArray_DATA(r2),
        <double*>np.PyArray_DATA(r3),
        <double*>np.PyArray_DATA(normal),
        <double*>np.PyArray_DATA(obs),

        <double*>np.PyArray_DATA(I),
        <double*>np.PyArray_DATA(Irho),
        )

    return I, Irho


def potint2(
        np.ndarray[np.float64_t, ndim=2] r1,
        np.ndarray[np.float64_t, ndim=2] r2,
        np.ndarray[np.float64_t, ndim=2] r3,
        np.ndarray[np.float64_t, ndim=2] normal,
        np.ndarray[np.float64_t, ndim=2] obs,
        ):
    """
        N
        r1 - 1 x 3 triangle vertex
        r2 - 1 x 3 triangle vertex
        r3 - 1 x 3 triangle vertex
        normal - 1 x 3 triangle normal
        obs - N x 3 matrix of observation points
        -------------------------------------------------------------
        Int - N x 3 matrix of gradient integrals Int = grad(Is(1/r))

        SP 2026
    """
    N = obs.shape[0]
    r1 = np.asfortranarray(r1, dtype=np.float64)
    r2 = np.asfortranarray(r2, dtype=np.float64)
    r3 = np.asfortranarray(r3, dtype=np.float64)
    normal = np.asfortranarray(normal, dtype=np.float64)
    obs = np.asfortranarray(obs, dtype=np.float64)

    Int = np.zeros((N, 3), dtype=np.float64, order="F")
    c_potint2(
        N,
        <double*>np.PyArray_DATA(r1),
        <double*>np.PyArray_DATA(r2),
        <double*>np.PyArray_DATA(r3),
        <double*>np.PyArray_DATA(normal),
        <double*>np.PyArray_DATA(obs),

        <double*>np.PyArray_DATA(Int),
        )
    return Int
