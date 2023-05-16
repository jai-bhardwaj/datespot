# Stage 1: Build Stage
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS build

# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for OpenMPI
ENV PATH=/usr/local/openmpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libatlas-base-dev \
        libcppunit-dev \
        libhdf5-dev \
        libjsoncpp-dev \
        libnetcdf-c++4-dev \
        libnetcdf-dev \
        libopenmpi-dev \
        pkg-config \
        python3 \
        software-properties-common \
        unzip \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Download and install cub library
WORKDIR /tmp
RUN wget -O cub.zip https://github.com/NVlabs/cub/archive/2.1.0.zip && \
    unzip cub.zip && \
    mv cub-2.1.0/cub /usr/local/include/ && \
    rm -rf cub.zip cub-2.1.0

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

# Set the PATH environment variable
ENV PATH=/opt/bin:${PATH}

# Copy only the necessary files from the build stage to reduce the final image size
COPY --from=build /usr/local/openmpi /usr/local/openmpi
COPY --from=build /build/bin /opt/bin

# Reduce image size by removing unnecessary packages and files
RUN apt-get update && \
    apt-get purge -y --auto-remove \
        build-essential \
        cmake \
        wget \
        unzip \
        pkg-config \
        software-properties-common && \
    find /usr/local -name "*.a" -delete && \
    find /usr/local -name "*.cmake" -delete && \
    find /usr/local -name "*.so" -type f -exec strip --strip-all {} + && \
    rm -rf /usr/local/cuda-11.4.0/{doc,doc-targets,extras,nsight*,nvvm,tools} && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y && \
    apt-get clean
