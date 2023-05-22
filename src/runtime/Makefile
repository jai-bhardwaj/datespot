SHELL := /bin/sh

# Set the default build directory
BUILD_DIR := build

# Set the source directory
SOURCES_DIR := .

# Include the common Makefile settings
include ../Makefile.inc

# Set the directories for object files and headers
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/runtime/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/runtime/cuda
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/runtime

# Create the necessary directories
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))
$(shell mkdir -p $(HEADERS_BUILD_DIR))

# Find all header files
HEADERS := $(wildcard *.h *.hpp)

# Find all source files
SOURCES := $(wildcard *.c *.cc *.cpp)

# Find all CUDA source files
CU_SOURCES := $(wildcard *.cu)

# Set the object file names
OBJECTS := $(patsubst %.cpp,$(OBJS_BUILD_DIR)/%.o,$(SOURCES))
CU_OBJECTS := $(patsubst %.cu,$(CU_OBJS_BUILD_DIR)/%.o,$(CU_SOURCES))

# Set the dependency file names
DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)

# Set the library file name
LIB_BUILD_DIR := $(BUILD_DIR)/lib
LIB_tensorhub := $(LIB_BUILD_DIR)/libtensorhub.a

# Set the linker flags
LDFLAGS ?= -fPIC

# Set the library and linker flags
LIBS := $(CU_LIBS) -L$(BUILD_DIR)/lib
LOAD_LIBS := $(CU_LOADLIBS) -ltensorhub_engine

# Set the compiler flags
CFLAGS := $(CU_INCLUDES)

# Set the CUDA compiler flags
CU_FLAGS := $(CU_INCLUDES) $($*.cu.CU_FLAGS)

# Set the CUDA dependency flags
CU_DEP_FLAGS := -E -std=c++11 -Xcompiler "-isystem $(CUDA_SYSTEM_INCLUDE_DIR) -MP,-MM"

# Set the target file
TARGET := $(LIB_BUILD_DIR)/libtensorhub_runtime.so

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJECTS) $(CU_OBJECTS)
	@echo "========== Creating libtensorhub_runtime.so =========="
	mkdir -p $(LIB_BUILD_DIR)
	$(CC) -shared $(LDFLAGS) $(LIBS) $(OBJECTS) $(CU_OBJECTS) -o $@ $(LOAD_LIBS)
	@echo "========== Copying runtime headers =========="
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

# Rule to build C++ object files
$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build CUDA object files
$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu GpuTypes.h
	$(NVCC) $(CU_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $($*.cu.CU_FLAGS) -c $< -o $@

# Rule to generate CUDA dependency files
$(CU_OBJS_BUILD_DIR)/%.d: $(SOURCES_DIR)/%.cu
	$(NVCC) $(CU_DEP_FLAGS) $(CU_INCLUDES) $< -o $@

# Include the dependency files
-include $(DEP)
-include $(CU_DEP)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Clean object files
distclean:
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)
