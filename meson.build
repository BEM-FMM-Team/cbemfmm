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
  c_args = ['/O2', '/openmp']
  link_args = ['/openmp']
elif host_machine.system() == 'darwin'
  # TODO may need more testing
  c_args = ['-O3', '-fopenmp', '-funroll-loops']
  link_args = ['-fopenmp']
else  # Linux
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
