from setuptools import setup, Extension
import numpy as np

# Define the extension module
module1 = Extension('tensorhub',
                    sources=['tensorhubmodule.cc', '../src/utils/cdl.cpp'],
                    include_dirs=[
                        np.get_include(),  # Include NumPy headers
                        '/usr/local/cuda/include',  # CUDA include directory
                        'B40C',  # Additional include directories
                        'B40C/KernelCommon',
                        '/usr/lib/openmpi/include',  # OpenMPI include directory
                        '/usr/include/jsoncpp',  # JSONCPP include directory
                        '../src/utils',  # Additional include directories
                        '../src/engine',
                        '/usr/include/cppunit',
                        '/usr/local/lib/python3.8/dist-packages/tensorflow/include'
                    ],
                    libraries=[
                        'tensorhub',  # Required libraries
                        'cudnn',
                        'curand',
                        'cublas',
                        'cudart',
                        'jsoncpp',
                        'netcdf',
                        'blas',
                        'dl',
                        'stdc++',
                        'netcdf_c++4',
                        'tensorflow_framework'
                    ],
                    library_dirs=[
                        '/usr/lib/x86_64-linux-gnu',  # Library directories
                        '/usr/lib/atlas-base',
                        '/usr/local/cuda/lib64',
                        '/usr/local/lib',
                        '../system-tensorhub/lib',
                        '/usr/local/lib/python3.8/dist-packages/tensorflow'
                    ],
                    language='c++17',  # Use C++17 language standard
                    extra_compile_args=['-DOMPI_SKIP_MPICXX']  # Additional compile arguments
                    )

# Setup configuration
setup(name='tensorhub',
      version='1.0',
      description='This is a package that links functions to Mojo.',
      ext_modules=[module1],  # Include the extension module
      setup_requires=['numpy'],  # NumPy is required during setup
      install_requires=['numpy', 'tensorflow'])  # NumPy and TensorFlow are required during installation

