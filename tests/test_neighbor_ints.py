import numpy as np
import pytest
from reference import GAUSS_RULES, facet_geometry, sphere_mesh
from scipy.sparse import csr_matrix

from cbemfmm import neighbor_ints_En, neighbor_ints_Pn, potint

P, t = sphere_mesh(162)
normals, center, area = facet_geometry(P, t)
T = len(t)
tu = t.astype(np.uintp)

# the rules are stored to 7-10 digits, so their weights only sum to 1 that well
WEIGHT_SUM_ERR = {
    1: 1e-13,
    3: 1e-13,
    4: 1e-13,
    6: 1e-8,
    7: 3e-7,
    9: 1e-13,
    13: 1e-8,
    25: 1e-10,
    48: 1e-13,
}
EDGE_POINTS = pytest.mark.xfail(
    strict=True,
    reason="the rule has points on facet edges, where the solid angle degenerates",
)


def solid_angles(P, t, gauss):
    """
    En with unit areas and zero normals is only the solid angle term:
    column n of 4pi * EC holds the gauss-averaged solid angle that each
    neighbor subtends from facet n, here with every facet as neighbor.
    """
    T = len(t)
    _, center, _ = facet_geometry(P, t)
    nbr = np.tile(np.arange(T, dtype=np.uintp), (T, 1))
    EC = neighbor_ints_En(
        P, t.astype(np.uintp), np.zeros((T, 3)), center, nbr, np.ones(T), gauss
    )
    return 4 * np.pi * np.asarray(EC.sum(axis=0)).ravel()


@pytest.mark.parametrize(
    "gauss",
    [
        pytest.param(g, marks=EDGE_POINTS) if g in (3, 6, 9) else g
        for g in WEIGHT_SUM_ERR
    ],
)
def test_En_closed_surface_solid_angle(gauss):
    # the rest of a closed surface subtends 2pi from any point on it
    omega = solid_angles(P, t, gauss)
    assert (
        np.abs(np.abs(omega) / (2 * np.pi) - 1).max() <= WEIGHT_SUM_ERR[gauss] + 1e-12
    )


@pytest.mark.parametrize("gauss", [7, 25, 48])
def test_En_inside_solid_angle(gauss):
    # a facet inside a closed surface sees all of it, 4pi
    inner = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]])
    P2 = np.vstack([P, inner])
    t2 = np.vstack([t, [len(P), len(P) + 1, len(P) + 2]])
    omega = solid_angles(P2, t2, gauss)[T]
    assert abs(abs(omega) / (4 * np.pi) - 1) <= WEIGHT_SUM_ERR[gauss] + 1e-12


def random_neighbors(M, seed=0):
    # every facet itself plus M - 1 distinct others, as a T x M array
    rng = np.random.default_rng(seed)
    others = [
        rng.choice(np.delete(np.arange(T), k), M - 1, replace=False) for k in range(T)
    ]
    return np.column_stack([np.arange(T), others])


def gauss_points(gauss):
    bary, w = GAUSS_RULES[gauss]
    return np.einsum("gk,tkx->tgx", bary, P[t]), w


def assemble(nbr, data):
    # data[n, m] belongs to row nbr[n, m], column n, duplicates summed
    jj = np.repeat(np.arange(T), nbr.shape[1])
    return csr_matrix((data.ravel(), (nbr.ravel(), jj)), shape=(T, T)).toarray()


@pytest.mark.parametrize("gauss", list(GAUSS_RULES))
def test_En_formula(gauss):
    """
    EC[nb, n] = (A_n / A_nb <Omega_nb>_n - IC) / 4pi with the solid angle
    of nb averaged over the gauss points of n (Van Oosterom and Strackee)
    and IC = -A_n (c_n - c_nb) . n_nb / |c_n - c_nb|^3, zero for nb == n.
    """
    nbr = random_neighbors(8)
    X, w = gauss_points(gauss)
    data = np.zeros(nbr.shape)
    for n in range(T):
        for m, nb in enumerate(nbr[n]):
            if nb == n:
                continue
            R = P[t[nb]][None] - X[n][:, None]
            r = np.linalg.norm(R, axis=2)
            num = np.einsum("gx,gx->g", R[:, 0], np.cross(R[:, 1], R[:, 2]))
            den = (
                r.prod(axis=1)
                + np.einsum("gx,gx->g", R[:, 0], R[:, 1]) * r[:, 2]
                + np.einsum("gx,gx->g", R[:, 0], R[:, 2]) * r[:, 1]
                + np.einsum("gx,gx->g", R[:, 1], R[:, 2]) * r[:, 0]
            )
            IE = w @ (2 * np.arctan2(num, den))
            d = center[n] - center[nb]
            IC = -area[n] * (d @ normals[nb]) / np.linalg.norm(d) ** 3
            data[n, m] = (IE * area[n] / area[nb] - IC) / (4 * np.pi)
    EC = neighbor_ints_En(P, tu, normals, center, nbr.astype(np.uintp), area, gauss)
    ref = assemble(nbr, data)
    assert np.abs(EC.toarray() - ref).max() <= 1e-13 * np.abs(ref).max()


@pytest.mark.parametrize("gauss", list(GAUSS_RULES))
def test_Pn_formula(gauss):
    """
    PC[nb, n] = (<int_n 1/r>_nb - A_n / |c_n - c_nb|) / 4pi with the
    potential of source n averaged over the gauss points of observer nb,
    the center term zero for nb == n. Neighbors are those of each source.
    """
    nbr = random_neighbors(8)
    X, w = gauss_points(gauss)
    data = np.zeros(nbr.shape)
    for n in range(T):
        obs = X[nbr[n]].reshape(-1, 3)
        I, _ = potint(*P[t[n]][:, None], normals[n][None], obs)
        d = np.linalg.norm(center[n] - center[nbr[n]], axis=1)
        approx = np.divide(area[n], d, out=np.zeros_like(d), where=nbr[n] != n)
        data[n] = (I.reshape(-1, len(w)) @ w - approx) / (4 * np.pi)
    PC = neighbor_ints_Pn(P, tu, normals, center, nbr.astype(np.uintp), area, gauss)
    ref = assemble(nbr, data)
    assert np.abs(PC.toarray() - ref).max() <= 1e-13 * np.abs(ref).max()
