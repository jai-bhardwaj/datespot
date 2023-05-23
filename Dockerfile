# Stage 1: Build Stage
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS build

# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

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
        libeigen3-dev \
        pkg-config \
        python3 \
        software-properties-common \
        unzip \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install cub library
WORKDIR /tmp
RUN wget -O cub.zip https://github.com/NVlabs/cub/archive/2.1.0.zip \
    && unzip cub.zip \
    && mv cub-2.1.0/cub /usr/local/include/ \
    && rm -rf cub.zip cub-2.1.0

# Create a build directory
WORKDIR /build

# Copy the necessary source code files to the build directory
COPY src Makefile /build/

# Set build flags and parallelize the build
ARG NUM_THREADS=8
RUN make -j${NUM_THREADS} CXXFLAGS="-O3 -march=native" install \
    && make clean

# Stage 2: Final Stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS final

# Set environment variables for OpenMPI
ENV PATH=/usr/local/openmpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Install necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenmpi3 \
    && rm -rf /var/lib/apt/lists/*

# Copy necessary files from the build stage
COPY --from=build /usr/local/openmpi /usr/local/openmpi
COPY --from=build /build/bin /opt/bin

COPY --from=build /usr/include/eigen3 /usr/include/eigen3

# Reduce image size by removing unnecessary packages and files
RUN apt-get purge -y --auto-remove \
        build-essential \
        cmake \
        wget \
        unzip \
        pkg-config \
        software-properties-common \
    && find /usr/local -name "*.a" -delete \
    && find /usr/local -name "*.cmake" -delete \
    && find /usr/local -name "*.so" -type f -exec strip --strip-all {} + \
    && rm -rf /var/lib/apt/lists/*

# Set the PATH environment variable
ENV PATH=/opt/bin:${PATH}

# Set the entrypoint
ENTRYPOINT ["/opt/bin/tensorhub"]

