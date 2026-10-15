# cbemfmm

C library companion to bem-fmm-python: the integrals over triangular facets
that the BEM-FMM solver needs near the singularity, in C with OpenMP and a
Cython wrapper.

| function | computes |
|---|---|
| `potint(r1, r2, r3, normal, obs)` | Is(1/r) and Is((y - p)/r) over one triangle at N points (Wilton et al. 1984) |
| `potint2(r1, r2, r3, normal, obs)` | grad Is(1/r) over one triangle at N points |
| `neighbor_ints_En(P, t, normal, center, neighbor, area, gauss)` | sparse E-field correction for the neighbors of every facet (solid angle, Van Oosterom and Strackee 1983) |
| `neighbor_ints_Pn(P, t, normal, center, neighbor, area, gauss)` | sparse potential correction for the neighbors of every facet |
| `volume_near_ints(points, ptr, idx, P, t, normal, center, area, c)` | near-field correction of the E-field at volume points, near facets given as CSR lists |

The docstrings give shapes and formulas. Inputs of any float or integer
type are accepted; C-ordered float64 and int64 arrays are used without a
copy. The normal of each triangle must be `cross(r2 - r1, r3 - r1)`
normalized.

## Build and test

Needs a C compiler; OpenMP is used when the compiler has it.

```
pip install .[test]
pytest tests
```

Wheels for Linux, macOS and Windows (CPython 3.11 to 3.14) are built and
tested by `.github/workflows/build_and_release.yml` on `v*` tags.
