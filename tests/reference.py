import numpy as np
from scipy.integrate import quad_vec
from scipy.spatial import ConvexHull


def unit_normal(r1, r2, r3):
    n = np.cross(r2 - r1, r3 - r1)
    return n / np.linalg.norm(n)


def triangle_integrals(r1, r2, r3, x, epsrel=1e-13):
    """
    I = int 1/R, Irho = int (y - p)/R and G = int (y - x)/R^3 over the
    triangle, with R = |y - x| and p the projection of x on its plane.

    The triangle is the signed sum of the triangles (p, a, b) over its
    edges. Each one is integrated in polar coordinates about p, the radial
    integrals in closed form and the angle by adaptive Gauss-Kronrod, so
    nothing here shares code or formulas with the edge-based kernels.
    G is None for in-plane x, where it is only a principal value.
    """
    n = unit_normal(r1, r2, r3)
    d = n @ (x - r1)
    p = x - d * n
    e1 = (r2 - r1) / np.linalg.norm(r2 - r1)
    e2 = np.cross(n, e1)
    scale = max(np.linalg.norm(r2 - r1), np.linalg.norm(r3 - r1))
    ad = abs(d)

    total = np.zeros(7)
    for a, b in ((r1, r2), (r2, r3), (r3, r1)):
        va, vb = a - p, b - p
        if min(np.linalg.norm(va), np.linalg.norm(vb)) < 1e-14 * scale:
            continue  # p at a vertex
        dth = np.arctan2(n @ np.cross(va, vb), va @ vb)
        if abs(dth) < 1e-15:
            continue  # p on the edge line
        ta = np.arctan2(va @ e2, va @ e1)
        edge = (vb - va) / np.linalg.norm(vb - va)
        perp = va - (va @ edge) * edge
        h = np.linalg.norm(perp)
        th0 = np.arctan2(perp @ e2, perp @ e1)

        def radial(theta):
            rm = h / np.cos(theta - th0)
            u = np.cos(theta) * e1 + np.sin(theta) * e2
            Rm = np.hypot(rm, d)
            i1 = Rm - ad  # int rho/R drho
            if ad > 0:
                lg = np.log((rm + Rm) / ad)
                i2 = 0.5 * (rm * Rm - d * d * lg)  # int rho^2/R
                i3 = 1 / ad - 1 / Rm  # int rho/R^3
                i4 = lg - rm / Rm  # int rho^2/R^3
            else:
                i2, i3, i4 = 0.5 * rm * rm, 0.0, 0.0
            return np.concatenate([[i1], u * i2, u * i4 - d * n * i3])

        total += quad_vec(radial, ta, ta + dth, epsabs=0, epsrel=epsrel, limit=400)[0]

    G = total[4:] if ad > 0 else None
    return total[0], total[1:4], G


def sphere_mesh(num_points, radius=0.08):
    """Closed triangle mesh of a sphere with outward normals, 2 * num_points - 4 facets."""
    k = np.arange(num_points) + 0.5
    z = 1 - 2 * k / num_points
    phi = np.pi * (1 + 5**0.5) * k
    rho = np.sqrt(1 - z * z)
    P = radius * np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])
    t = ConvexHull(P).simplices
    v = P[t]
    inward = (
        np.einsum("tx,tx->t", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 0])
        < 0
    )
    t[inward] = t[inward][:, ::-1]
    return P, t


def facet_geometry(P, t):
    """normals, centers and areas of the facets"""
    v = P[t]
    cr = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    return cr / (2 * area[:, None]), v.mean(axis=1), area


# barycentric points (rows) and weights of the rules as stored in coeffs.h
GAUSS_RULES = {
    1: (np.full((1, 3), 1 / 3), np.array([1.0])),
    4: (
        np.array(
            [[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        ),
        np.array([-0.5625, 25 / 48, 25 / 48, 25 / 48]),
    ),
}
