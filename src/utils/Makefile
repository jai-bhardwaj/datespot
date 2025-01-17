SHELL=/bin/sh
VPATH=


include ../Makefile.inc

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/utils

OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/cuda
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_BUILD_DIR)/%.o)
OBJS := $(addprefix $(OBJS_BUILD_DIR)/, $(addsuffix .o, $(basename $(HEADERS))))

LIB_BUILD_DIR := $(BUILD_DIR)/lib

LIB_tensorhub :=$(BUILD_DIR)/lib/libtensorhub.a

BIN_BUILD_DIR := $(BUILD_DIR)/bin
$(shell mkdir -p $(BIN_BUILD_DIR))

INCLUDES = \
    $(CU_INCLUDES) \
    -I../engine

LIBS = \
    $(CU_LIBS) \

LOAD_LIBS = $(CU_LOADLIBS)

LDFLAGS ?= -fPIC -Wl,--whole-archive $(LIB_tensorhub) -Wl,--no-whole-archive

EXECUTABLES := \
	$(BIN_BUILD_DIR)/generateNetCDF \
	$(BIN_BUILD_DIR)/train \
	$(BIN_BUILD_DIR)/predict \
	$(BIN_BUILD_DIR)/encoder

all: $(EXECUTABLES) $(LIB_BUILD_DIR)/libtensorhub_utils.so

$(LIB_tensorhub):
	cd ../engine && make

$(LIB_BUILD_DIR)/libtensorhub_utils.so: $(OBJS)
	$(info ========== Creating libtensorhub_utils.so ==========)
	mkdir -p $(BUILD_DIR)/lib
	$(CC) -shared $(LDFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)
	$(info ========== Copying/utils headers ==========)
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

$(BIN_BUILD_DIR)/generateNetCDF: $(OBJS) $(LIB_tensorhub) $(OBJS_BUILD_DIR)/NetCDFGenerator.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_BUILD_DIR)/train: $(OBJS) $(LIB_tensorhub) $(OBJS_BUILD_DIR)/Train.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_BUILD_DIR)/encoder: $(OBJS) $(LIB_tensorhub) $(OBJS_BUILD_DIR)/main.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

$(BIN_BUILD_DIR)/predict: $(OBJS) $(LIB_tensorhub) $(OBJS_BUILD_DIR)/Predict.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $@ $(LOAD_LIBS)

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(BIN_BUILD_DIR) $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)/libtensorhub_utils.so

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)

$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

TST_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/utils/tests

$(BIN_BUILD_DIR)/TestGPU: $(OBJS) $(LIB_tensorhub) $(TST_OBJS_BUILD_DIR)/TestDune.o
	$(LOAD) $(LOADFLAGS) $(LIBS) $^ -o $(BIN_BUILD_DIR)/$@ $(LOAD_LIBS) -l:libcppunit.a

$(TST_OBJS_BUILD_DIR)/TestDune.o: ../../../../tests/gputests/TestDune.cpp
		mkdir -p $(TST_OBJS_BUILD_DIR)
		$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
