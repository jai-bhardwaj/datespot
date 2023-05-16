# Stage 1: Build Stage
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS build

# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libcppunit-dev \
        libatlas-base-dev \
        pkg-config \
        python3 \
        software-properties-common \
        unzip \
        wget \
        cmake \
        libopenmpi-dev \
        libjsoncpp-dev \
        libhdf5-dev \
        zlib1g-dev \
        libnetcdf-dev \
        libnetcdf-c++4-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install cub library
WORKDIR /usr/local/include
RUN wget -O cub.zip https://github.com/NVlabs/cub/archive/2.1.0.zip && \
    unzip cub.zip && \
    mv cub-2.1.0/cub . && \
    rm -rf cub.zip cub-2.1.0

# Set environment variables
ENV PATH=/usr/local/openmpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Create a build directory
WORKDIR /build

# Copy only the necessary source code files to the build directory
COPY src /build/src
COPY Makefile /build/Makefile

# Set build flags and parallelize the build
ARG NUM_THREADS=8
RUN make -j${NUM_THREADS} CXXFLAGS="-O3 -march=native" install && \
    make clean

# Stage 2: Final Stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS final

# Copy only the necessary files from the build stage to reduce the final image size
COPY --from=build /usr/local/openmpi /usr/local/openmpi
COPY --from=build /build/bin /opt/bin

# Set the PATH environment variable
ENV PATH=/opt/bin:${PATH}

# Remove build dependencies and cleanup
RUN apt-get update && \
    apt-get purge -y --auto-remove \
        build-essential \
        cmake \
        wget \
        unzip \
        pkg-config \
        software-properties-common && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/include/cub && \
    find /usr/local -name "*.a" -delete && \
    find /usr/local -name "*.cmake" -delete && \
    find /usr/local -name "*.so" -type f -exec strip --strip-all {} + && \
    rm -rf /usr/local/cuda-11.4.0/doc && \
    rm -rf /usr/local/cuda-11.4.0/doc-targets && \
    rm -rf /usr/local/cuda-11.4.0/extras && \
    rm -rf /usr/local/cuda-11.4.0/nsight* && \
    rm -rf /usr/local/cuda-11.4.0/nvvm && \
    rm -rf /usr/local/cuda-11.4.0/tools

# Reduce image size by removing unnecessary packages and files
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
