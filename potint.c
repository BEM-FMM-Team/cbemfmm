/*
 * potint.c - Calculation of the single potential integrals (1/r)
 * Follows the method described in: Wilton DR, Rao SM, Glisson AW,
 * Schaubert DH, Al-Bundak OM, Butler CM. Potential integrals for
 * uniform  and linear source distribution on polygonal and
 * polyhedral domains.
 * IEEE Trans Antennas Propag 1984;AP-32 (3):276–281 - uses last
 * formula in Eq. (5) on page 279
 *
 * Guillermo Nunez Ponasso (2026)
 */

#include "cbemfmm.h"

#include <math.h>
#include <omp.h>

void single_layer_potint(
    // clang-format off
    const double r1_x, const double r1_y, const double r1_z,
    const double r2_x, const double r2_y, const double r2_z,
    const double r3_x, const double r3_y, const double r3_z,
    const double n_x, const double n_y, const double n_z, // normal to triangle r
    const double o_x, const double o_y, const double o_z, // observation point coords
    // outputs I and Irho
    double *I,
    double *Irho_x, double *Irho_y, double *Irho_z
    // clang-format on
) {

  // create l coordinates
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

  double normal_abs_l1_inv, normal_abs_l2_inv, normal_abs_l3_inv;
  normal_abs_l1_inv = 1.0 / norm3(dx_21, dy_21, dz_21);
  normal_abs_l2_inv = 1.0 / norm3(dx_31, dy_31, dz_31);
  normal_abs_l3_inv = 1.0 / norm3(dx_32, dy_32, dz_32);

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

  // create unit normal to the edges of the triangle
  double u1_x, u1_y, u1_z;
  double u2_x, u2_y, u2_z;
  double u3_x, u3_y, u3_z;

  cross(l1_x, l1_y, l1_z, n_x, n_y, n_z, &u1_x, &u1_y,
        &u1_z); // u_1 = cross(l1, n)

  cross(l2_x, l2_y, l2_z, -n_x, -n_y, -n_z, &u2_x, &u2_y,
        &u2_z); // u_2 = -cross(l2, n) = cross(l2,-n)

  cross(l3_x, l3_y, l3_z, n_x, n_y, n_z, &u3_x, &u3_y,
        &u3_z); // u_3 = cross(l1, n)

  // create projection vector of the observation point
  double ndot;
  ndot = dot(o_x, o_y, o_z, n_x, n_y, n_z);

  double p_x, p_y, p_z;
  p_x = o_x - ndot * n_x;
  p_y = o_y - ndot * n_y;
  p_z = o_z - ndot * n_z;

  // calculation of the analytic fomula

  // using arrays and a small fixed-length for loop should be taken
  // care of by the compiler using loop unrolling and scalar replacement.

  // storing variables in arrays
  // r+ and r- coordinate
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
  double logdiv;

  int count = 0, c3, c3_1;

  *I = 0; // initialize I, Irho
  *Irho_x = 0;
  *Irho_y = 0;
  *Irho_z = 0;
  for (int k = 0; k < 3; k++) {
    c3 = 3 * count;
    c3_1 = 3 * (count + 1);

    dpp_x = o_x - r[c3_1];
    dpp_y = o_y - r[c3_1 + 1];
    dpp_z = o_z - r[c3_1 + 2];
    distance_obs = fabs(dot(n_x, n_y, n_z, dpp_x, dpp_y, dpp_z));

    // calculate p+ and p-
    d1 = dot(n_x, n_y, n_z, r[c3], r[c3 + 1], r[c3 + 2]);
    d2 = dot(n_x, n_y, n_z, r[c3_1], r[c3_1 + 1], r[c3_1 + 2]);

    pplus_x = r[c3] - n_x * d1;
    pplus_y = r[c3 + 1] - n_y * d1;
    pplus_z = r[c3 + 2] - n_z * d1;

    pminus_x = r[c3_1] - n_x * d2;
    pminus_y = r[c3_1 + 1] - n_y * d2;
    pminus_z = r[c3_1 + 2] - n_z * d2;

    // calculate l+ and l-
    lplus = dot(l[3 * k], l[3 * k + 1], l[3 * k + 2], pplus_x - p_x,
                pplus_y - p_y, pplus_z - p_z);
    lminus = dot(l[3 * k], l[3 * k + 1], l[3 * k + 2], pminus_x - p_x,
                 pminus_y - p_y, pminus_z - p_z);

    // perpendicular distance from vector to edge
    P0 = fabs(dot(u[3 * k], u[3 * k + 1], u[3 * k + 2], pminus_x - p_x,
                  pminus_y - p_y, pminus_z - p_z));
    // Contribution is zero when the projection point is on the edge
    // (Wilton et. al. 1984, p.279)
    // P0 multiplier: if P0 < TOL then we regard P0 = 0.
    P0_mult = (P0 < TOL) ? 0 : 1.0 / P0;

    // distances to l+ and l- from projection vector
    PPLUS = sqrt(P0 * P0 + lplus * lplus);
    PMINUS = sqrt(P0 * P0 + lminus * lminus);

    // vector containing line P0 measured
    PHAT_x = (pminus_x - p_x - lminus * l[3 * k]) * P0_mult;
    PHAT_y = (pminus_y - p_y - lminus * l[3 * k + 1]) * P0_mult;
    PHAT_z = (pminus_z - p_z - lminus * l[3 * k + 2]) * P0_mult;

    // Distances to l+ and l- from observation point
    RPLUS = sqrt(PPLUS * PPLUS + distance_obs * distance_obs);
    RMINUS = sqrt(PMINUS * PMINUS + distance_obs * distance_obs);
    R0 = sqrt(P0 * P0 + distance_obs * distance_obs);

    // a value of one term of the analytic sum 1/R
    logdiv = log((RPLUS + lplus) / (RMINUS + lminus));
    D1 = P0 * logdiv;
    D2 = distance_obs * atan2(P0 * lplus, R0 * R0 + distance_obs * RPLUS);
    D3 = distance_obs * atan2(P0 * lminus, R0 * R0 + distance_obs * RMINUS);

    dotPHATu =
        dot(PHAT_x, PHAT_y, PHAT_z, u[3 * k], u[3 * k + 1], u[3 * k + 2]);
    *I = *I + (D1 - D2 + D3) * dotPHATu;

    // a value of one term of the analytic sum RHO/R
    D1 = R0 * R0 * logdiv;
    D2 = lplus * RPLUS - lminus * RMINUS;
    D3 = D1 + D2;

    *Irho_x = *Irho_x + D3 * u[3 * k];
    *Irho_y = *Irho_y + D3 * u[3 * k + 1];
    *Irho_z = *Irho_z + D3 * u[3 * k + 2];

    count = count + 2;
  }
  *Irho_x = 0.5 * (*Irho_x);
  *Irho_y = 0.5 * (*Irho_y);
  *Irho_z = 0.5 * (*Irho_z);
}

/*
 * gateway function callable by Python
 * N
 * r1 - 1 x 3 triangle vertex
 * r2 - 1 x 3 triangle vertex
 * r3 - 1 x 3 triangle vertex
 * normal - 1 x 3 triangle normal
 * obs - N x 3 matrix of observation points
 * -------------------------------------------------------------
 * I - N x 1 vector of neighbor integrals I = Is(1/r)
 * Irho - N x 3 matrix of neighbor integrals I = Is(vec(r)/r)
 */
void c_potint(const size_t N, const double *restrict r1,
              const double *restrict r2, const double *restrict r3,
              const double *restrict normal, const double *restrict obs,
              double *restrict I, double *restrict Irho) {
  long k;
#pragma omp parallel for schedule(static)
  for (k = 0; k < (long)N; k++) {
    single_layer_potint(r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1],
                        r3[2], normal[0], normal[1], normal[2], obs[k],
                        obs[k + N], obs[k + 2 * N], &(I[k]), &(Irho[k]),
                        &(Irho[k + N]), &(Irho[k + 2 * N]));
  }
}
