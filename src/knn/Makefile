SHELL := /bin/sh

# Set the build directory
BUILD_DIR := build

# Compiler and flags
CC := g++
NVCC := nvcc
CFLAGS := -Wall -Wextra -O2
CU_FLAGS := -arch=sm_30

# Directories
VPATH :=
SOURCES_DIR := .
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/knn
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/knn/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/knn/cuda
LIB_BUILD_DIR := $(BUILD_DIR)/lib

# Include directories
CU_INCLUDES := -I../utils
CUDA_SYSTEM_INCLUDE_DIR := /usr/local/cuda/include

# Files
SOURCES := $(wildcard *.c *.cc *.cpp)
CU_SOURCES := $(wildcard *.cu)
HEADERS := $(wildcard *.h *.hpp)
OBJECTS := $(patsubst $(SOURCES_DIR)/%.cpp,$(OBJS_BUILD_DIR)/%.o,$(SOURCES))
CU_OBJECTS := $(patsubst $(SOURCES_DIR)/%.cu,$(CU_OBJS_BUILD_DIR)/%.o,$(CU_SOURCES))
DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)
LIB := $(LIB_BUILD_DIR)/libtensorhub_knn.so

# Create necessary directories
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

.PHONY: all clean distclean

all: $(LIB)

$(LIB): $(OBJECTS) $(CU_OBJECTS)
	@echo "========== Creating $@ =========="
	mkdir -p $(LIB_BUILD_DIR)
	$(CC) -shared $(LDFLAGS) $(CU_LIBS) $(OBJECTS) $(CU_OBJECTS) -o $@ $(CU_LOADLIBS)
	@echo "========== Copying knn headers =========="
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)/libtensorhub_knn.a

distclean: clean

$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CU_INCLUDES) -MMD -c $< -o $@

$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu
	$(NVCC) $(CU_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $(CU_INCLUDES) -MMD -c $< -o $@

-include $(DEP)
-include $(CU_DEP)
