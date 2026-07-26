cdef extern from "cbemfmm.h":
    void cneighbor_ints_En(
        const double *P,
        const size_t *t,
        const double *normal,
        const double *center,
        const size_t *neighbor,
        const double *area,
        size_t N,
        size_t T,
        size_t M,
        int gauss,
        double *IE,
        double *IC
    )

    void cneighbor_ints_Pn(
        const double *P,
        const size_t *t,
        const double *normal,
        const double *center,
        const size_t *neighbor,
        const double *area,
        size_t N,
        size_t T,
        size_t M,
        int gauss,
        double *IP,
        double *IPC
    )

    void c_potint(
        const size_t N,
        const double *r1,
        const double *r2,
        const double *r3,
        const double *normal,
        const double *obs,
        double *I,
        double *Irho
        )

    void c_potint2(
        const size_t N,
        const double *r1,
        const double *r2,
        const double *r3,
        const double *normal,
        const double *obs,
        double *Int
        )
