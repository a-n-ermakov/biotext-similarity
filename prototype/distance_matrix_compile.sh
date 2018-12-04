source activate p36_dis
# python distance_matrix_setup.py build_ext --inplace
cython distance_matrix.pyx
mkdir build
gcc -pthread -B $HOME/miniconda3/envs/p36_dis/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I$HOME/miniconda3/envs/p36_dis/include/python3.6m -c distance_matrix.c -o build/distance_matrix.o -I$HOME/miniconda3/envs/p36_dis/lib/python3.6/site-packages/numpy/core/include
gcc -shared build/distance_matrix.o -o build/distance_matrix.so
source deactivate p36_dis