MODULE_ROOT := $(shell pwd)

BUILD_DIR ?= $(MODULE_ROOT)/../build
BIN_DIR := $(BUILD_DIR)/tst/bin
TEST_SOURCES_DIR := system
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/tst

SOURCES := $(shell find $(TEST_SOURCES_DIR) -type f -name '*.cpp')
OBJECTS := $(patsubst $(TEST_SOURCES_DIR)/%.cpp,$(OBJS_BUILD_DIR)/%.o,$(SOURCES))
DEP := $(OBJECTS:.o=.d)

tensorhub_SRC_DIR := ../src
LIB_tensorhub := $(BUILD_DIR)/lib/libtensorhub.a
LIB_tensorhub_UTILS := $(BUILD_DIR)/lib/libtensorhub_utils.so

INCLUDES := \
	-isystem /usr/local/cuda/include \
	-isystem /usr/include/jsoncpp \
	-isystem /usr/lib/openmpi/include \
	-isystem /usr/lib/openmpi/include/openmpi \
	-I../src \
	-I/path/to/gtest/include

LIB := \
	$(BUILD_DIR)/lib \
	/usr/local/cuda/lib64 \
	/usr/lib/openmpi/lib

LIBS := $(addprefix -L,$(LIB))

LLIB := \
	cudnn \
	curand \
	cublas \
	cudart \
	jsoncpp \
	netcdf \
	netcdf_c++4 \
	blas \
	dl \
	stdc++ \
	mpi_cxx \
	mpi \
	gtest \ # Google Test library
	tensorhub_utils

LOAD_LIBS := $(addprefix -l,$(LLIB))

CC := g++
CFLAGS ?= -Wall -std=c++20 -pthread
LDFLAGS ?= -Wl,-rpath=/usr/local/cuda/lib64

.PHONY: all clean run-tests

all: $(BIN_DIR)/unittests

$(LIB_tensorhub):
	$(MAKE) -C $(tensorhub_SRC_DIR)/engine

$(LIB_tensorhub_UTILS):
	$(MAKE) -C $(tensorhub_SRC_DIR)/utils

$(BIN_DIR)/unittests: $(LIB_tensorhub) $(LIB_tensorhub_UTILS) $(OBJECTS)
	@echo "========== Building unittests ============="
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) unittests.cpp $^ $(LIB_tensorhub) -o $@ $(LOAD_LIBS) -lgtest

run-tests: $(BIN_DIR)/unittests
	@echo "========== Running unittests ============="
	LD_LIBRARY_PATH=$(BUILD_DIR)/lib $(BIN_DIR)/unittests

clean:
	rm -rf $(BIN_DIR)

$(OBJS_BUILD_DIR)/%.o: $(TEST_SOURCES_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP $(INCLUDES) -c $< -o $@

