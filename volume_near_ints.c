/*
 * volume_near_ints.c - Near-field correction of the electric field at
 * volume points. For every point the center-point approximation of each
 * nearby facet is replaced by the exact integral grad(Is(1/r)).
 *
 * Shawn Pande (2026)
 */

#include "cbemfmm.h"

/*
 * gateway function callable by Python
 * Np - number of points
 * points - Np x 3 observation points
 * ptr, idx - near facets of point p are idx[ptr[p]] to idx[ptr[p + 1] - 1]
 * P, t, normal, center, area - mesh, as for cneighbor_ints_En
 * c - T x 1 surface charge density
 * ---------------------------------------
 * dE - Np x 3, -1/(4pi) * sum over near facets m of
 *      c[m] * (grad(Is_m(1/r)) - A_m (C_m - x) / |C_m - x|^3)
 */
void c_volume_near_ints(size_t Np, const double *restrict points,
                        const size_t *restrict ptr, const size_t *restrict idx,
                        const double *restrict P, const size_t *restrict t,
                        const double *restrict normal,
                        const double *restrict center,
                        const double *restrict area, const double *restrict c,
                        double *restrict dE) {
  const double scale = -1.0 / (4.0 * acos(-1.0));
  long p;
  // points differ a lot in their number of near facets
#pragma omp parallel for schedule(dynamic, 16)
  for (p = 0; p < (long)Np; p++) {
    double x = points[3 * p], y = points[3 * p + 1], z = points[3 * p + 2];
    double E_x = 0, E_y = 0, E_z = 0;
    for (size_t j = ptr[p]; j < ptr[p + 1]; j++) {
      size_t m = idx[j];
      const double *r1 = P + 3 * t[3 * m];
      const double *r2 = P + 3 * t[3 * m + 1];
      const double *r3 = P + 3 * t[3 * m + 2];
      double I_x, I_y, I_z;
      grad_potint(r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2],
                  normal[3 * m], normal[3 * m + 1], normal[3 * m + 2], x, y, z,
                  &I_x, &I_y, &I_z);

      double d_x = center[3 * m] - x;
      double d_y = center[3 * m + 1] - y;
      double d_z = center[3 * m + 2] - z;
      double D = norm3(d_x, d_y, d_z);
      double AD3 = area[m] / (D * D * D);

      E_x += c[m] * (I_x - AD3 * d_x);
      E_y += c[m] * (I_y - AD3 * d_y);
      E_z += c[m] * (I_z - AD3 * d_z);
    }
    dE[3 * p] = scale * E_x;
    dE[3 * p + 1] = scale * E_y;
    dE[3 * p + 2] = scale * E_z;
  }
}
