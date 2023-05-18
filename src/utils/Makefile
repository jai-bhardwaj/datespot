SHELL := /bin/sh
VPATH :=

include ../Makefile.inc

# Source and build directories
SOURCES_DIR := .
BUILD_DIR := build

# Header files
HEADERS := $(wildcard *.h *.hpp)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/utils

# Object file directories
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/cuda
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

# Source files
SOURCES := $(wildcard *.c *.cc *.cpp)
CU_SOURCES := $(wildcard *.cu)

# Object files
OBJECTS := $(SOURCES:%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_BUILD_DIR)/%.o)
OBJS := $(addprefix $(OBJS_BUILD_DIR)/,$(addsuffix .o,$(basename $(HEADERS))))

# Library build directory
LIB_BUILD_DIR := $(BUILD_DIR)/lib

# Libraries
LIB_tensorhub := $(BUILD_DIR)/lib/libtensorhub.a

# Binary build directory
BIN_BUILD_DIR := $(BUILD_DIR)/bin
$(shell mkdir -p $(BIN_BUILD_DIR))

# Include directories and libraries
INCLUDES := $(CU_INCLUDES) -I../engine
LIBS := $(CU_LIBS)
LOAD_LIBS := $(CU_LOADLIBS)

# Linker flags
LDFLAGS ?= -fPIC -Wl,--whole-archive $(LIB_tensorhub) -Wl,--no-whole-archive

# Executables
EXECUTABLES := $(patsubst %,$(BIN_BUILD_DIR)/%,$(notdir $(basename $(SOURCES))))

.PHONY: all clean distclean

# Default target
all: $(EXECUTABLES) $(LIB_BUILD_DIR)/libtensorhub_utils.so

# Build the tensorhub library
$(LIB_tensorhub):
	$(MAKE) -C ../engine

# Build the libtensorhub_utils.so library
$(LIB_BUILD_DIR)/libtensorhub_utils.so: $(OBJS)
	@echo "========== Creating libtensorhub_utils.so =========="
	mkdir -p $(LIB_BUILD_DIR)
	$(CC) -shared $(LDFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)
	@echo "========== Copying utils headers =========="
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

# Build the executables
$(BIN_BUILD_DIR)/%: $(OBJS) $(LIB_tensorhub) $(OBJS_BUILD_DIR)/%.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

# Clean the project
clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(BUILD_DIR)

# Clean and remove all generated files
distclean: clean
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)

# Compile C++ source files
$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Create the test object directory and compile the test object file
TST_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/tst
$(BIN_BUILD_DIR)/TestGPU: $(OBJS) $(LIB_tensorhub) $(TST_OBJS_BUILD_DIR)/TestDune.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS) -l:libcppunit.a

# Compile the test object file
$(TST_OBJS_BUILD_DIR)/TestDune.o: ../../../../tst/gputests/TestDune.cpp | $(TST_OBJS_BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Create the test object directory
$(TST_OBJS_BUILD_DIR):
	mkdir -p $@
