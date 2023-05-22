SHELL := /bin/sh

# Directories
VPATH :=
BUILD_DIR := build
SOURCES_DIR := .
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/engine
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/engine/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/engine/cuda
LIB_BUILD_DIR := $(BUILD_DIR)/lib

# Compiler and flags
CC := gcc
CXX := g++
NVCC := nvcc
CFLAGS := -Wall -Wextra -O2
CXXFLAGS := -Wall -Wextra -O2 -std=c++20
LDFLAGS := -L$(LIB_BUILD_DIR)
CU_INCLUDES := -I../utils
CU_LIBS := -lcudart
CU_FLAGS := -arch=sm_30

# Source files
HEADERS := $(wildcard *.h *.hpp)
SOURCES := $(wildcard *.c *.cc *.cpp)
CU_SOURCES := $(wildcard *.cu)

# Object files
OBJECTS := $(patsubst %.cpp,$(OBJS_BUILD_DIR)/%.o,$(SOURCES))
CU_OBJECTS := $(patsubst %.cu,$(CU_OBJS_BUILD_DIR)/%.o,$(CU_SOURCES))

# Dependency files
DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)

# Targets
TARGET_LIB := $(LIB_BUILD_DIR)/libtensorhub.a
TARGET_SHARED_LIB := $(LIB_BUILD_DIR)/libtensorhub_engine.so

.PHONY: all clean distclean

all: $(TARGET_LIB) $(TARGET_SHARED_LIB)

$(TARGET_LIB): $(OBJECTS) $(CU_OBJECTS)
	@mkdir -p $(LIB_BUILD_DIR)
	@echo "========== Creating $@ =========="
	@ar rcs $@ $^
	@echo "========== Copying engine headers =========="
	@mkdir -p $(HEADERS_BUILD_DIR)
	@cp $(HEADERS) $(HEADERS_BUILD_DIR)

$(TARGET_SHARED_LIB): $(OBJECTS) $(CU_OBJECTS)
	@mkdir -p $(LIB_BUILD_DIR)
	@echo "========== Creating $@ =========="
	@$(CXX) -shared $(LDFLAGS) $(CU_LIBS) $^ -o $@

clean:
	@echo "========== Cleaning up =========="
	@rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(HEADERS_BUILD_DIR) $(TARGET_LIB)

distclean: clean
	@echo "========== Cleaning all =========="
	@rm -rf $(BUILD_DIR)

$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	@mkdir -p $(OBJS_BUILD_DIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) $(CU_INCLUDES) -c $< -o $@

$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu GpuTypes.h $(CU_OBJS_BUILD_DIR)/%.d
	@mkdir -p $(CU_OBJS_BUILD_DIR)
	@echo "Compiling $<"
	@$(NVCC) $(CU_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $(CU_INCLUDES) $($*.cu.CU_FLAGS) -c $< -o $@

$(CU_OBJS_BUILD_DIR)/%.d: $(SOURCES_DIR)/%.cu
	@mkdir -p $(CU_OBJS_BUILD_DIR)
	@echo "Generating dependency $@"
	@$(NVCC) -E -std=c++20 -Xcompiler "-isystem $(CUDA_SYSTEM_INCLUDE_DIR) -MP,-MM" $(CU_INCLUDES) $< -o $@

-include $(DEP)
-include $(CU_DEP)

