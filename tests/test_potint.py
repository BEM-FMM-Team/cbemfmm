import numpy as np
import pytest
from reference import triangle_integrals, unit_normal

from cbemfmm import potint, potint2

# scalene triangle of a few mm in general position, in metres like a head mesh
r1 = np.array([0.010, 0.020, 0.030])
r2 = np.array([0.0127, 0.0208, 0.0302])
r3 = np.array([0.0109, 0.0231, 0.0297])
n = unit_normal(r1, r2, r3)
c = (r1 + r2 + r3) / 3
L = np.mean([np.linalg.norm(r2 - r1), np.linalg.norm(r3 - r2), np.linalg.norm(r1 - r3)])
e = (r2 - r1) / np.linalg.norm(r2 - r1)
s = np.cross(n, e)

CASES = {
    "far": c + 20 * L * (0.3 * n + 0.8 * e + 0.5 * s),
    "mid": c + 3 * L * (0.6 * n - 0.6 * e + 0.5 * s),
    "above centroid 1e-1": c + 0.1 * L * n,
    "above centroid 1e-3": c + 1e-3 * L * n,
    "below inside": 0.6 * r1 + 0.3 * r2 + 0.1 * r3 - 1e-2 * L * n,
    "above edge": 0.5 * (r1 + r2) + 1e-2 * L * n,
    "above vertex": r3 + 1e-2 * L * n,
    "above edge 1e-6": 0.5 * (r1 + r2) + 1e-6 * L * n,
    "above edge line": r1 + 1.5 * (r2 - r1) + 1e-3 * L * n,
    "far along edge line": r1 + 30 * (r2 - r1) + 1e-4 * L * n,
    "in-plane outside": c - 1.5 * L * s + 0.2 * L * e,
    "in-plane far": c + 5 * L * s,
    "in-plane centroid": c,
    "in-plane inside": 0.2 * r1 + 0.7 * r2 + 0.1 * r3,
    "in-plane on edge": 0.3 * r1 + 0.7 * r2,
    "in-plane vertex": r1,
    "in-plane edge line beyond r2": r1 + 1.5 * (r2 - r1),
    "in-plane edge line beyond r1": r1 - 0.5 * (r2 - r1),
}
NAMES = list(CASES)
OBS = np.array(list(CASES.values()))
REF = {k: triangle_integrals(r1, r2, r3, x) for k, x in CASES.items()}

# one call for all points, so the per-point indexing is checked too
tri = [v[None] for v in (r1, r2, r3, n)]
I, Irho = potint(*tri, OBS)
GRAD = potint2(*tri, OBS)

POTINT_NAN = pytest.mark.xfail(
    strict=True,
    reason="log((R+ + l+)/(R- + l-)) is 0, inf or 0/0 on the edge line, P0 * inf is NaN",
)
EDGE_LOG = pytest.mark.xfail(
    strict=True,
    reason="cancellation in log((R+ + l+)/(R- + l-)) and the nudge of off-plane points",
)


def mark(names, marker, keep=NAMES):
    return [pytest.param(k, marks=marker) if k in names else k for k in keep]


@pytest.mark.parametrize(
    "name",
    mark(
        {"in-plane on edge", "in-plane vertex", "in-plane edge line beyond r2"},
        POTINT_NAN,
    ),
)
def test_potint(name):
    I_ref, Irho_ref, _ = REF[name]
    k = NAMES.index(name)
    assert abs(I[k, 0] - I_ref) <= 1e-12 * abs(I_ref)
    assert np.linalg.norm(Irho[k] - Irho_ref) <= 1e-12 * np.linalg.norm(Irho_ref)


@pytest.mark.parametrize(
    "name",
    mark(
        {
            "above edge",
            "above vertex",
            "above edge 1e-6",
            "above edge line",
            "far along edge line",
        },
        EDGE_LOG,
        [k for k in NAMES if not k.startswith("in-plane")],
    ),
)
def test_potint2(name):
    _, _, G_ref = REF[name]
    # inside the edge band the point is still nudged by EDGE_FACTOR * L
    rtol = 1e-6 if name == "above edge 1e-6" else 1e-12
    k = NAMES.index(name)
    assert np.linalg.norm(GRAD[k] - G_ref) <= rtol * np.linalg.norm(G_ref)
