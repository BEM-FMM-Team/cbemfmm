/*
 * neighbor_ints_Pn.c - Calculation of the integral of 1/r
 * for all neighbors of all triangles of a mesh.
 * Follows the method described in: Wilton DR, Rao SM, Glisson AW,
 * Schaubert DH, Al-Bundak OM, Butler CM. Potential integrals for
 * uniform  and linear source distribution on polygonal and
 * polyhedral domains.
 * IEEE Trans Antennas Propag 1984;AP-32 (3):276–281 - uses last
 * formula in Eq. (5) on page 279
 * Guillermo Nunez Ponasso (2026)
 */
#include "coeffs.h"
#include "cbemfmm.h"

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

void cneighbor_ints_Pn(const double *restrict P, const size_t *restrict t,
                        const double *restrict normal,
                        const double *restrict center,
                        const size_t *restrict neighbor,
                        const double *restrict area, size_t N, size_t T,
                        size_t M, int gauss, double *restrict IP,
                        double *restrict IPC) {
  // === retrieve the gaussian cubature points ===
  const double *coeff, *weight;

  switch (gauss) {
  case 1:
    coeff = COEFFS(1);
    weight = WEIGHTS(1);
    break;
  case 3:
    coeff = COEFFS(3);
    weight = WEIGHTS(3);
    break;
  case 4:
    coeff = COEFFS(4);
    weight = WEIGHTS(4);
    break;
  case 6:
    coeff = COEFFS(6);
    weight = WEIGHTS(6);
    break;
  case 7:
    coeff = COEFFS(7);
    weight = WEIGHTS(7);
    break;
  case 9:
    coeff = COEFFS(9);
    weight = WEIGHTS(9);
    break;
  case 13:
    coeff = COEFFS(13);
    weight = WEIGHTS(13);
    break;
  case 25:
    coeff = COEFFS(25);
    weight = WEIGHTS(25);
    break;
  case 48:
    coeff = COEFFS(48);
    weight = WEIGHTS(48);
    break;
  default:
    coeff = COEFFS(25);
    weight = WEIGHTS(25);
    break;
  }
  // === main loop ===
  long n;
#pragma omp parallel for schedule(static)
  for (n = 0; n < T; n++) {
    size_t g, m;
    size_t v1 = t[n];
    size_t v2 = t[n + T];
    size_t v3 = t[n + 2 * T];

    double Pn1_x = P[v1];
    double Pn1_y = P[v1 + N];
    double Pn1_z = P[v1 + 2 * N];

    double Pn2_x = P[v2];
    double Pn2_y = P[v2 + N];
    double Pn2_z = P[v2 + 2 * N];

    double Pn3_x = P[v3];
    double Pn3_y = P[v3 + N];
    double Pn3_z = P[v3 + 2 * N];

    double n_x = normal[n];
    double n_y = normal[n + T];
    double n_z = normal[n + 2 * T];

    for (m = 0; m < M; m++) {
      size_t nb = neighbor[n + m * T];

      double Pmg_x[BUFF], Pmg_y[BUFF], Pmg_z[BUFF];

      size_t u1 = t[nb];
      size_t u2 = t[nb + T];
      size_t u3 = t[nb + 2 * T];

      for (g = 0; g < gauss; g++) {
        Pmg_x[g] = coeff[g] * P[u1] + coeff[g + gauss] * P[u2] +
                   coeff[g + 2 * gauss] * P[u3];
        Pmg_y[g] = coeff[g] * P[u1 + N] + coeff[g + gauss] * P[u2 + N] +
                   coeff[g + 2 * gauss] * P[u3 + N];
        Pmg_z[g] = coeff[g] * P[u1 + 2 * N] + coeff[g + gauss] * P[u2 + 2 * N] +
                   coeff[g + 2 * gauss] * P[u3 + 2 * N];
      }
      double IP_sum = 0.0;
      double I_temp;
      double Irho_x, Irho_y, Irho_z;
      for (g = 0; g < gauss; g++) {
        single_layer_potint(Pn1_x, Pn1_y, Pn1_z, Pn2_x, Pn2_y, Pn2_z, Pn3_x,
                            Pn3_y, Pn3_z, n_x, n_y, n_z, Pmg_x[g], Pmg_y[g],
                            Pmg_z[g], &I_temp, &Irho_x, &Irho_y, &Irho_z);
        IP_sum += weight[g] * I_temp;
      }
      IP[n + m * T] = IP_sum;

      if (nb == n)
        continue; // no self-terms
      double Dnm = dist(center[n], center[n + T], center[n + 2 * T], center[nb],
                        center[nb + T], center[nb + 2 * T]);
      double ADnm = area[n] / Dnm;
      IPC[n + m * T] = ADnm;
    }
  }
}
