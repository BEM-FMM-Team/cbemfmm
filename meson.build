project('cbemfmm', 'c',
  version: '0.1.0',
  license: 'MIT',
  default_options: [
    'optimization=3'
  ]
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

cc = meson.get_compiler('c')

if host_machine.system() == 'windows'
  if cc.get_id() == 'msvc'
    c_args = ['/O2', '/openmp']
    link_args = []
  else
    # mingw/clang on Windows
    c_args = ['-O3', '-fopenmp', '-funroll-loops']
    link_args = ['-fopenmp']
  endif
elif host_machine.system() == 'darwin'
  if cc.get_id() == 'clang'
    # Apple clang: use Xpreprocessor + libomp (not plain -fopenmp)
    omp_prefix = run_command('brew', ['--prefix', 'libomp'], check: false).stdout().strip()
    if omp_prefix != ''
      c_args = ['-O3', '-Xpreprocessor', '-fopenmp', '-funroll-loops', '-I' + omp_prefix + '/include']
      link_args = ['-L' + omp_prefix + '/lib', '-lomp', '-Wl,-rpath,' + omp_prefix + '/lib']
    else
      # Fallback: build without OpenMP if libomp is unavailable
      c_args = ['-O3', '-funroll-loops']
      link_args = []
    endif
  else
    c_args = ['-O3', '-fopenmp', '-funroll-loops']
    link_args = ['-fopenmp']
  endif
else
  c_args = ['-O3', '-fopenmp', '-funroll-loops']
  link_args = ['-fopenmp']
endif

c_sources = [
  'neighbor_ints_En.c',
  'potint2.c',
  'potint.c',
]

cython_prog = find_program('cython', required: true)

gen_cython = custom_target(
  'cbemfmm_cython',
  output: 'cbemfmm.c',
  input: 'cbemfmm.pyx',
  command: [cython_prog, '-3', '@INPUT@', '-o', '@OUTPUT@']
)

py.extension_module(
  'cbemfmm',
  [gen_cython, c_sources],
  include_directories: inc_np,
  dependencies: py_dep,
  c_args: c_args,
  link_args: link_args,
  install: true
)
