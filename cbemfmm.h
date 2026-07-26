#ifndef cbemfmm_H
#define cbemfmm_H

#include <math.h>
#include <stddef.h>

void cneighbor_ints_En(const double *restrict P, const size_t *restrict t,
                        const double *restrict normal,
                        const double *restrict center,
                        const size_t *restrict neighbor,
                        const double *restrict area, size_t N, size_t T,
                        size_t M, int gauss, double *restrict IE,
                        double *restrict IC);

void cneighbor_ints_Pn(const double *restrict P, const size_t *restrict t,
                        const double *restrict normal,
                        const double *restrict center,
                        const size_t *restrict neighbor,
                        const double *restrict area, size_t N, size_t T,
                        size_t M, int gauss, double *restrict IP,
                        double *restrict IPC);

void c_potint(const size_t N, const double *restrict r1,
              const double *restrict r2, const double *restrict r3,
              const double *restrict normal, const double *restrict obs,
              double *restrict r_I, double *restrict r_Irho);

void c_potint2(const size_t N, const double *restrict r1,
               const double *restrict r2, const double *restrict r3,
               const double *restrict normal, const double *restrict obs,
               double *restrict Int);

#define TOL 1e-10
#define EDGE_FACTOR 1e-6

// buffer size for cubature points
// WARNING: This needs to be changed if larger cubatures are implemented
#define BUFF 128

inline double dot(const double u_x, const double u_y, const double u_z,
                  const double v_x, const double v_y, const double v_z) {
  return u_x * v_x + u_y * v_y + u_z * v_z;
}

inline double norm3(const double u_x, const double u_y, const double u_z) {
  return sqrt(dot(u_x, u_y, u_z, u_x, u_y, u_z));
}

inline void cross(const double u_x, const double u_y, const double u_z,
                  const double v_x, const double v_y, const double v_z,
                  double *w_x, double *w_y, double *w_z) {
  *w_x = u_y * v_z - u_z * v_y;
  *w_y = u_z * v_x - u_x * v_z;
  *w_z = u_x * v_y - u_y * v_x;
}

// Compute the triple product of three 3D vectors
inline double triple_product(double u1, double u2, double u3, double v1,
                             double v2, double v3, double w1, double w2,
                             double w3) {
  double a, b, c;
  a = v2 * w3 - v3 * w2;
  b = u2 * w3 - u3 * w2;
  c = u2 * v3 - u3 * v2;
  return u1 * a - v1 * b + w1 * c;
}

inline double dist(double u1, double u2, double u3, double v1, double v2,
                   double v3) {
  return norm3(u1 - v1, u2 - v2, u3 - v3);
}

#endif
