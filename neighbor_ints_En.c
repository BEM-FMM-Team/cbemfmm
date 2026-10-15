/*******************************************************************
 * neighbor_ints_En.c - Calculation of the integral of n*grad(1/r) *
 * for all neighbors of all triangles of a mesh. Here we use the   *
 * solid angle approach with Gaussian quadrature. With computation *
 * of the solid angle as described in Van Oosterom & Strackee 1983 *
 * Guillermo Nunez Ponasso (2026)                                  *
 *******************************************************************/
#include "cbemfmm.h"
#include "coeffs.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

// solid angle subtended by triangle of vertices u,v,w from observation point t
static double solid_angle(double u1, double u2, double u3, double v1, double v2,
                          double v3, double w1, double w2, double w3, double t1,
                          double t2, double t3) {

  double R1x, R1y, R1z, R2x, R2y, R2z, R3x, R3y, R3z;
  R1x = u1 - t1;
  R1y = u2 - t2;
  R1z = u3 - t3; // displacements
  R2x = v1 - t1;
  R2y = v2 - t2;
  R2z = v3 - t3;
  R3x = w1 - t1;
  R3y = w2 - t2;
  R3z = w3 - t3;

  double r1, r2, r3;
  r1 = sqrt(R1x * R1x + R1y * R1y + R1z * R1z); // norms of displacements
  r2 = sqrt(R2x * R2x + R2y * R2y + R2z * R2z);
  r3 = sqrt(R3x * R3x + R3y * R3y + R3z * R3z);

  double d12, d13, d23;
  d12 = R1x * R2x + R1y * R2y + R1z * R2z; // pairwise dot products
  d13 = R1x * R3x + R1y * R3y + R1z * R3z;
  d23 = R2x * R3x + R2y * R3y + R2z * R3z;

  double N =
      triple_product(R1x, R1y, R1z, R2x, R2y, R2z, R3x, R3y, R3z); // numerator

  double D = r1 * r2 * r3 + d12 * r3 + d13 * r2 + d23 * r1; // denominator

  return 2.0 * atan2(N, D);
}

void cneighbor_ints_En(const double *restrict P, const size_t *restrict t,
                       const double *restrict normal,
                       const double *restrict center,
                       const size_t *restrict neighbor,
                       const double *restrict area, size_t T, size_t M,
                       int gauss, double *restrict IE, double *restrict IC) {
  // === retrieve the gaussian cubature points ===
  const double *coeff, *weight;
  // int *indexF;

  // clang-format off
  switch (gauss) {
    case 1: coeff = COEFFS(1); weight = WEIGHTS(1); break;
    case 3: coeff = COEFFS(3); weight = WEIGHTS(3); break;
    case 4: coeff = COEFFS(4); weight = WEIGHTS(4); break;
    case 6: coeff = COEFFS(6); weight = WEIGHTS(6); break;
    case 7: coeff = COEFFS(7); weight = WEIGHTS(7); break;
    case 9: coeff = COEFFS(9); weight = WEIGHTS(9); break;
    case 13: coeff = COEFFS(13); weight = WEIGHTS(13); break;
    case 25: coeff = COEFFS(25); weight = WEIGHTS(25); break;
    case 48: coeff = COEFFS(48); weight = WEIGHTS(48); break;
    default: coeff = COEFFS(25); weight = WEIGHTS(25); break;
  }
  // clang-format on

  // === main loop ===
  long n;
#pragma omp parallel for schedule(static)
  for (n = 0; n < T; n++) {
    size_t g, m;
    double Png_x[BUFF], Png_y[BUFF], Png_z[BUFF];
    for (g = 0; g < gauss; g++) {
      size_t v1 = t[3 * n];
      size_t v2 = t[3 * n + 1];
      size_t v3 = t[3 * n + 2];

      Png_x[g] = coeff[g] * P[3 * v1] + coeff[g + gauss] * P[3 * v2] +
                 coeff[g + 2 * gauss] * P[3 * v3];
      Png_y[g] = coeff[g] * P[3 * v1 + 1] + coeff[g + gauss] * P[3 * v2 + 1] +
                 coeff[g + 2 * gauss] * P[3 * v3 + 1];
      Png_z[g] = coeff[g] * P[3 * v1 + 2] + coeff[g + gauss] * P[3 * v2 + 2] +
                 coeff[g + 2 * gauss] * P[3 * v3 + 2];
    }

    // calculate solid angle contribution for all neighbors of tn
    for (m = 0; m < M; m++) {
      size_t nb = neighbor[n * M + m]; // neighbor ID
      if (nb == n)
        continue; // no self-term

      size_t v1 = t[3 * nb], v2 = t[3 * nb + 1], v3 = t[3 * nb + 2];
      double Pm1_x = P[3 * v1], Pm1_y = P[3 * v1 + 1], Pm1_z = P[3 * v1 + 2];
      double Pm2_x = P[3 * v2], Pm2_y = P[3 * v2 + 1], Pm2_z = P[3 * v2 + 2];
      double Pm3_x = P[3 * v3], Pm3_y = P[3 * v3 + 1], Pm3_z = P[3 * v3 + 2];

      double IE_sum = 0.0;
      for (g = 0; g < gauss; g++) {
        IE_sum += weight[g] * solid_angle(Pm1_x, Pm1_y, Pm1_z, Pm2_x, Pm2_y,
                                          Pm2_z, Pm3_x, Pm3_y, Pm3_z, Png_x[g],
                                          Png_y[g], Png_z[g]);
      }
      IE[n * M + m] = IE_sum;

      // calculate centerpoint interactions
      double dnm_x = center[3 * n] - center[3 * nb];
      double dnm_y = center[3 * n + 1] - center[3 * nb + 1];
      double dnm_z = center[3 * n + 2] - center[3 * nb + 2];
      double Dnm = norm3(dnm_x, dnm_y, dnm_z);
      double ADnm3 = -area[n] / (Dnm * Dnm * Dnm);

      IC[n * M + m] =
          dot(dnm_x * ADnm3, dnm_y * ADnm3, dnm_z * ADnm3, normal[3 * nb],
              normal[3 * nb + 1], normal[3 * nb + 2]);
    }
  }
}
