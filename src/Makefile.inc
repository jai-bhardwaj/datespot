# Compiler and tools
CC = mpiCC
NVCC = nvcc
LOAD = mpiCC

# Build directory
BUILD_DIR ?= $(dir $(realpath $(firstword $(MAKEFILE_LIST))))../../../build

# CUDA system include directory
CUDA_SYSTEM_INCLUDE_DIR := $(dir $(shell which $(NVCC)))../target/x86_64/include

# Debug mode flag
DEBUG ?= 0

ifeq ($(DEBUG), 1)
  $(info ************  DEBUG mode ************)
  CFLAGS = -g -O0 -Wall
  CU_FLAGS = -g -O0 --device-debug --generate-line-info
else
  $(info ************  RELEASE mode ************)
  CFLAGS = -O3
  CU_FLAGS = -O3
endif

# Common flags for C/C++ compilation
CFLAGS += -std=c++20 -fPIC -DOMPI_SKIP_MPICXX -MMD -MP

# Common flags for CUDA compilation
CU_FLAGS += -std=c++20 --compiler-options="-fPIC" -use_fast_math --ptxas-options="-v" \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_52,code=sm_52 \
            -gencode arch=compute_30,code=sm_30 \
            -DOMPI_SKIP_MPICXX

# Include directories
INCLUDES = -I/usr/local/include \
           -isystem /usr/local/cuda/include \
           -isystem /usr/lib/openmpi/include \
           -isystem /usr/include/jsoncpp \
           -IB40C \
           -IB40C/KernelCommon \
           -I$(BUILD_DIR)/include

# Library directories
LIB_DIRS = -L/usr/lib/atlas-base \
           -L/usr/local/cuda/lib64 \
           -L/usr/local/lib/

# Libraries to link
LIBS = -lcudnn -lcurand -lcublas -lcudart -ljsoncpp -lnetcdf_c++4 -lnetcdf -lblas -ldl -lstdc++

# Target executable
TARGET = your_executable

# Source files
SRCS = your_source_files.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Dependency files
DEPS = $(SRCS:.cpp=.d)

.PHONY: all clean

all: $(TARGET)

# Linking step
$(TARGET): $(OBJS)
	$(LOAD) $(OBJS) $(LIB_DIRS) $(LIBS) -o $(TARGET)

# Compilation step for C++ source files
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Compilation step for CUDA source files
%.o: %.cu
	$(NVCC) $(CU_FLAGS) $(INCLUDES) -c $< -o $@

# Include dependency files
-include $(DEPS)

clean:
	@rm -f $(OBJS) $(DEPS) $(TARGET)
