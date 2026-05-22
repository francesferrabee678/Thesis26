#!/bin/bash
#SBATCH --job-name=ME_Build
#SBATCH --partition=testing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-04:00:00
#SBATCH --output=/extra/fferrabe/Thesis26/lexicon3d/logs/eval-%j.out

set -e

source /home/fferrabe/miniconda3/etc/profile.d/conda.sh
conda activate m3drefclip

# 1. ENSURE COMPILER IS PRESENT
conda install -y -c conda-forge --force-reinstall gxx_linux-64=11 gcc_linux-64=11 sysroot_linux-64=2.17 openblas
pip install ninja

# 2. SET COMPILER AND PATHS
export CXX=$(ls $CONDA_PREFIX/bin/*g++ | head -n 1)
export CC=$(ls $CONDA_PREFIX/bin/*gcc | head -n 1)
export PATH=$CONDA_PREFIX/bin:$PATH

# Point CUDA_HOME and Includes to the conda targets
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export CUDA_INCLUDE_DIRS=$CONDA_PREFIX/targets/x86_64-linux/include
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4 

# Critical Flags for modern CUDA + Minkowski compatibility
export NVCCFLAGS="-DNVTX_DISABLE -Xcompiler -Wno-deprecated-declarations -Xcompiler -fpermissive"
export CXXFLAGS="-DNVTX_DISABLE -DTHRUST_IGNORE_CUB_VERSION_CHECK -Wno-deprecated-declarations -fpermissive"

echo "Using C compiler: $CC"
echo "Using C++ compiler: $CXX"


# Updated Flags to be more "forgiving" of deprecated Thrust code
export NVCCFLAGS="-DNVTX_DISABLE -Xcompiler -Wno-deprecated-declarations -Xcompiler -fpermissive -Wno-deprecated-gpu-targets"
export CXXFLAGS="-DNVTX_DISABLE -DTHRUST_IGNORE_CUB_VERSION_CHECK -Wno-deprecated-declarations -fpermissive"

# 3. PREPARE SOURCE AND APPLY PATCHES
pip uninstall -y MinkowskiEngine || true
rm -rf /tmp/MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine
cd /tmp/MinkowskiEngine

# A. Surgical Stub for NVTX
cat <<EOF > src/3rdparty/cudf/detail/nvtx/nvtx3.hpp
#pragma once
#define CUDF_FUNC_RANGE()
#define PUSH_RANGE(name, color_id)
#define POP_RANGE()
EOF
cp src/3rdparty/cudf/detail/nvtx/nvtx3.hpp src/3rdparty/cudf/detail/nvtx/ranges.hpp

# B. Fix missing Thrust algorithms & Policy headers
# We inject these into the CU file and the manager to ensure algorithms are visible
sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu
sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_manager.cu
sed -i '1i #include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh

# C. Resolve __to_address ambiguity (The Nuclear Option for GCC 11)
# We define a macro that forces the compiler to resolve to the global namespace
# This targets the specific collision in shared_ptr_base.h
export NVCCFLAGS="$NVCCFLAGS -D__to_address=::std::__to_address"

# D. Patch coordinate_map_gpu.cuh to handle shared_ptr assignment
# This adds a namespace qualifier directly to the source where the error triggers
sed -i 's/std::shared_ptr/::std::shared_ptr/g' src/coordinate_map_gpu.cuh

# E. Precision Strip for Thrust Deprecations
find src \( -name "*.cuh" -o -name "*.hpp" \) -exec sed -i 's/public thrust::unary_function<[^>]*>//g' {} +
find src \( -name "*.cuh" -o -name "*.hpp" \) -exec sed -i 's/public thrust::binary_function<[^>]*>//g' {} +

# 4. BUILD AND INSTALL
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

cd -
rm -rf /tmp/MinkowskiEngine

echo "MinkowskiEngine installation complete!"