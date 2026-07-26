/*
 * potint2.c - Calculation of the gradient potential integrals grad(1/r)
 * The integrals are not divided by the area.
 *
 * Guillermo Nunez Ponasso (2026)
 * Shawn Pande (2026)
 */

#include "cbemfmm.h"
#include <omp.h>

void grad_potint(const double r1_x, const double r1_y, const double r1_z,
                 const double r2_x, const double r2_y, const double r2_z,
                 const double r3_x, const double r3_y, const double r3_z,
                 const double n_x, const double n_y, const double n_z,
                 double o_x, double o_y, double o_z, double *Int_x,
                 double *Int_y, double *Int_z) {
  double dx_21, dy_21, dz_21;
  double dx_31, dy_31, dz_31;
  double dx_32, dy_32, dz_32;

  dx_21 = r2_x - r1_x;
  dy_21 = r2_y - r1_y;
  dz_21 = r2_z - r1_z;
  dx_31 = r3_x - r1_x;
  dy_31 = r3_y - r1_y;
  dz_31 = r3_z - r1_z;
  dx_32 = r3_x - r2_x;
  dy_32 = r3_y - r2_y;
  dz_32 = r3_z - r2_z;

  double len1, len2, len3;
  len1 = norm3(dx_21, dy_21, dz_21);
  len2 = norm3(dx_31, dy_31, dz_31);
  len3 = norm3(dx_32, dy_32, dz_32);

  double normal_abs_l1_inv, normal_abs_l2_inv, normal_abs_l3_inv;
  normal_abs_l1_inv = 1.0 / len1;
  normal_abs_l2_inv = 1.0 / len2;
  normal_abs_l3_inv = 1.0 / len3;

  double l1_x, l1_y, l1_z;
  double l2_x, l2_y, l2_z;
  double l3_x, l3_y, l3_z;

  l1_x = dx_21 * normal_abs_l1_inv;
  l1_y = dy_21 * normal_abs_l1_inv;
  l1_z = dz_21 * normal_abs_l1_inv;

  l2_x = dx_31 * normal_abs_l2_inv;
  l2_y = dy_31 * normal_abs_l2_inv;
  l2_z = dz_31 * normal_abs_l2_inv;

  l3_x = dx_32 * normal_abs_l3_inv;
  l3_y = dy_32 * normal_abs_l3_inv;
  l3_z = dz_32 * normal_abs_l3_inv;

  double u1_x, u1_y, u1_z;
  double u2_x, u2_y, u2_z;
  double u3_x, u3_y, u3_z;

  cross(l1_x, l1_y, l1_z, n_x, n_y, n_z, &u1_x, &u1_y, &u1_z);
  cross(l2_x, l2_y, l2_z, -n_x, -n_y, -n_z, &u2_x, &u2_y, &u2_z);
  cross(l3_x, l3_y, l3_z, n_x, n_y, n_z, &u3_x, &u3_y, &u3_z);

  double ndot;
  double p_x, p_y, p_z;

  ndot = dot(o_x, o_y, o_z, n_x, n_y, n_z);
  p_x = o_x - ndot * n_x;
  p_y = o_y - ndot * n_y;
  p_z = o_z - ndot * n_z;

  /* midpoints of the three edges, used to test whether the projection
     of the observation point lies on an edge or its extension */
  double mid1_x, mid1_y, mid1_z;
  double mid2_x, mid2_y, mid2_z;
  double mid3_x, mid3_y, mid3_z;

  mid1_x = 0.5 * (r1_x + r2_x);
  mid1_y = 0.5 * (r1_y + r2_y);
  mid1_z = 0.5 * (r1_z + r2_z);

  mid2_x = 0.5 * (r1_x + r3_x);
  mid2_y = 0.5 * (r1_y + r3_y);
  mid2_z = 0.5 * (r1_z + r3_z);

  mid3_x = 0.5 * (r2_x + r3_x);
  mid3_y = 0.5 * (r2_y + r3_y);
  mid3_z = 0.5 * (r2_z + r3_z);

  double check1, check2, check3;
  check1 = dot(mid1_x - p_x, mid1_y - p_y, mid1_z - p_z, u1_x, u1_y, u1_z);
  check2 = dot(mid2_x - p_x, mid2_y - p_y, mid2_z - p_z, u2_x, u2_y, u2_z);
  check3 = dot(mid3_x - p_x, mid3_y - p_y, mid3_z - p_z, u3_x, u3_y, u3_z);

  double factor1, factor2, factor3;
  factor1 = EDGE_FACTOR * len1;
  factor2 = EDGE_FACTOR * len2;
  factor3 = EDGE_FACTOR * len3;

  /* nudge the observation point off the edge line if its projection
     falls too close to one of the three edges */
  if (fabs(check1) < factor1) {
    o_x -= factor1 * u1_x;
    o_y -= factor1 * u1_y;
    o_z -= factor1 * u1_z;
  }
  if (fabs(check2) < factor2) {
    o_x -= factor2 * u2_x;
    o_y -= factor2 * u2_y;
    o_z -= factor2 * u2_z;
  }
  if (fabs(check3) < factor3) {
    o_x -= factor3 * u3_x;
    o_y -= factor3 * u3_y;
    o_z -= factor3 * u3_z;
  }

  /* recompute the projection point with the (possibly moved) observation point
   */
  ndot = dot(o_x, o_y, o_z, n_x, n_y, n_z);
  p_x = o_x - ndot * n_x;
  p_y = o_y - ndot * n_y;
  p_z = o_z - ndot * n_z;

  double r[18] = {r2_x, r2_y, r2_z, r1_x, r1_y, r1_z, r3_x, r3_y, r3_z,
                  r1_x, r1_y, r1_z, r3_x, r3_y, r3_z, r2_x, r2_y, r2_z};

  double u[9] = {u1_x, u1_y, u1_z, u2_x, u2_y, u2_z, u3_x, u3_y, u3_z};
  double l[9] = {l1_x, l1_y, l1_z, l2_x, l2_y, l2_z, l3_x, l3_y, l3_z};

  double dpp_x, dpp_y, dpp_z;
  double distance_obs;

  double d1, d2;

  double pplus_x, pplus_y, pplus_z;
  double pminus_x, pminus_y, pminus_z;

  double lplus, lminus;
  double P0, P0_mult;
  double PPLUS, PMINUS;
  double PHAT_x, PHAT_y, PHAT_z;
  double RPLUS, RMINUS, R0;
  double D1, D2, D3;
  double dotPHATu;

  double I_x, I_y, I_z;
  double Beta;

  int count = 0, c3, c3_1;

  I_x = 0;
  I_y = 0;
  I_z = 0;
  Beta = 0;

  for (int k = 0; k < 3; k++) {
    c3 = 3 * count;
    c3_1 = 3 * (count + 1);

    dpp_x = o_x - r[c3_1];
    dpp_y = o_y - r[c3_1 + 1];
    dpp_z = o_z - r[c3_1 + 2];
    distance_obs = dot(n_x, n_y, n_z, dpp_x, dpp_y, dpp_z);

    d1 = dot(n_x, n_y, n_z, r[c3], r[c3 + 1], r[c3 + 2]);
    d2 = dot(n_x, n_y, n_z, r[c3_1], r[c3_1 + 1], r[c3_1 + 2]);

    pplus_x = r[c3] - n_x * d1;
    pplus_y = r[c3 + 1] - n_y * d1;
    pplus_z = r[c3 + 2] - n_z * d1;

    pminus_x = r[c3_1] - n_x * d2;
    pminus_y = r[c3_1 + 1] - n_y * d2;
    pminus_z = r[c3_1 + 2] - n_z * d2;

    lplus = dot(l[3 * k], l[3 * k + 1], l[3 * k + 2], pplus_x - p_x,
                pplus_y - p_y, pplus_z - p_z);
    lminus = dot(l[3 * k], l[3 * k + 1], l[3 * k + 2], pminus_x - p_x,
                 pminus_y - p_y, pminus_z - p_z);

    P0 = fabs(dot(u[3 * k], u[3 * k + 1], u[3 * k + 2], pminus_x - p_x,
                  pminus_y - p_y, pminus_z - p_z));
    P0_mult = (P0 < TOL) ? 0 : 1.0 / P0;

    PPLUS = sqrt(P0 * P0 + lplus * lplus);
    PMINUS = sqrt(P0 * P0 + lminus * lminus);

    PHAT_x = (pminus_x - p_x - lminus * l[3 * k]) * P0_mult;
    PHAT_y = (pminus_y - p_y - lminus * l[3 * k + 1]) * P0_mult;
    PHAT_z = (pminus_z - p_z - lminus * l[3 * k + 2]) * P0_mult;

    RPLUS = sqrt(PPLUS * PPLUS + distance_obs * distance_obs);
    RMINUS = sqrt(PMINUS * PMINUS + distance_obs * distance_obs);
    R0 = sqrt(P0 * P0 + distance_obs * distance_obs);

    D1 = atan2(P0 * lplus, R0 * R0 + fabs(distance_obs) * RPLUS);
    D2 = atan2(P0 * lminus, R0 * R0 + fabs(distance_obs) * RMINUS);
    D3 = log((RPLUS + lplus) / (RMINUS + lminus));

    dotPHATu =
        dot(PHAT_x, PHAT_y, PHAT_z, u[3 * k], u[3 * k + 1], u[3 * k + 2]);

    Beta = Beta + dotPHATu * (D1 - D2);

    I_x = I_x + D3 * u[3 * k];
    I_y = I_y + D3 * u[3 * k + 1];
    I_z = I_z + D3 * u[3 * k + 2];

    count = count + 2;
  }

  double sign_val;
  sign_val = (distance_obs > 0) - (distance_obs < 0);

  *Int_x = -n_x * sign_val * Beta - I_x;
  *Int_y = -n_y * sign_val * Beta - I_y;
  *Int_z = -n_z * sign_val * Beta - I_z;

  if (isnan(*Int_x) || isinf(*Int_x))
    *Int_x = 0;
  if (isnan(*Int_y) || isinf(*Int_y))
    *Int_y = 0;
  if (isnan(*Int_z) || isinf(*Int_z))
    *Int_z = 0;
}

/*
 * gateway function callable by Python
 * N
 * r1 - 1 x 3 triangle vertex
 * r2 - 1 x 3 triangle vertex
 * r3 - 1 x 3 triangle vertex
 * normal - 1 x 3 triangle normal
 * obs - N x 3 matrix of observation points
 * ---------------------------------------
 * Int - N x 3 matrix of gradient integrals Int = grad(Is(1/r))
 */
void c_potint2(const size_t N, const double *restrict r1,
               const double *restrict r2, const double *restrict r3,
               const double *restrict normal, const double *restrict obs,
               double *restrict Int) {
  long k;
#pragma omp parallel for schedule(static)
  for (k = 0; k < (long)N; k++) {
    grad_potint(r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2],
                normal[0], normal[1], normal[2], obs[k], obs[k + N],
                obs[k + 2 * N], &(Int[k]), &(Int[k + N]), &(Int[k + 2 * N]));
  }
}
