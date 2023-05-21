from setuptools import setup, Extension
import numpy as np

# Define the extension module
module1 = Extension(
    'tensorhub',
    sources=['module.cc', '../src/utils/cdl.cpp'],
    include_dirs=[
        np.get_include(),
        '/usr/local/cuda/include',
        'B40C',
        'B40C/KernelCommon',
        '/usr/lib/openmpi/include',
        '/usr/include/jsoncpp',
        '../src/utils',
        '../src/engine',
        '/usr/include/cppunit',
        '/usr/local/lib/python3.8/dist-packages/tensorflow/include'
    ],
    libraries=[
        'tensorhub',
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
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/atlas-base',
        '/usr/local/cuda/lib64',
        '/usr/local/lib',
        '../system-tensorhub/lib',
        '/usr/local/lib/python3.8/dist-packages/tensorflow'
    ],
    language='c++20',
    extra_compile_args=['-DOMPI_SKIP_MPICXX']
)

# Setup configuration
setup(
    name='tensorhub',
    version='1.0',
    description='This is a package that links functions to Mojo.',
    ext_modules=[module1],
    setup_requires=['numpy'],
    install_requires=['numpy', 'tensorflow'],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 5 - Staging',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache-2.0 License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Documentation': 'https://github.com/dimske-sys/tensorhub/wiki',
        'Source': 'https://github.com/dimske-sys/tensorhub',
        'Bug Tracker': 'https://github.com/dimske-sys/tensorhub/issues',
    },
)

