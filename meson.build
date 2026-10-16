project('cbemfmm', 'c',
  version: '0.2.0',
  license: 'GPL-3.0-only',
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
# meson's openmp lookup on apple clang always takes homebrew's libomp, so on
# darwin link the libomp CPPFLAGS and LDFLAGS name
if host_machine.system() == 'darwin'
  omp = declare_dependency(
    compile_args: ['-Xpreprocessor', '-fopenmp'],
    link_args: ['-lomp'],
  )
else
  omp = dependency('openmp', required: false)
endif

c_sources = [
  'neighbor_ints_En.c',
  'neighbor_ints_Pn.c',
  'potint2.c',
  'potint.c',
  'volume_near_ints.c',
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
  dependencies: [py_dep, omp],
  c_args: cc.get_supported_arguments('-funroll-loops'),
  install: true
)
