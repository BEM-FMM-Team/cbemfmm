import numpy as np
import pytest
from reference import facet_geometry, sphere_mesh
from scipy.spatial import cKDTree

from cbemfmm import potint2, volume_near_ints

P, t = sphere_mesh(162)
normals, center, area = facet_geometry(P, t)
T = len(t)
rng = np.random.default_rng(1)
c = rng.normal(size=T)
# points in a shell around the surface (radius 0.08), many of them near facets
x = rng.normal(size=(500, 3))
points = x / np.linalg.norm(x, axis=1)[:, None] * rng.uniform(0.07, 0.09, size=(500, 1))
R = 3 * np.mean(np.sqrt(area))


def loop_reference():
    # the per-facet loop of bem-fmm-python's volume_field_electric
    E = np.zeros_like(points)
    tree = cKDTree(points)
    for m in range(T):
        index = np.asarray(tree.query_ball_point(center[m], R), dtype=np.int64)
        if index.size == 0:
            continue
        temp = center[m] - points[index]
        dist = np.sqrt(np.sum(temp * temp, axis=1))
        I = area[m] * temp / dist[:, None] ** 3
        E[index] -= -c[m] * I / (4 * np.pi)
        I = potint2(*P[t[m]][:, None], normals[m][None], points[index])
        E[index] += -c[m] * I / (4 * np.pi)
    return E


def near_lists():
    near = cKDTree(center).query_ball_point(points, R)
    ptr = np.cumsum([0] + [len(n) for n in near])
    return ptr, np.concatenate(near).astype(np.int64)


def test_matches_loop():
    ptr, idx = near_lists()
    assert idx.size > 10 * len(points)
    dE = volume_near_ints(points, ptr, idx, P, t, normals, center, area, c)
    ref = loop_reference()
    assert np.abs(dE - ref).max() <= 1e-12 * np.abs(ref).max()


def test_bad_ptr_raises():
    ptr, idx = near_lists()
    ptr[3], ptr[4] = ptr[4], ptr[3]
    with pytest.raises(ValueError):
        volume_near_ints(points, ptr, idx, P, t, normals, center, area, c)
