# Compiler and flags
CC := mpiCC
CFLAGS := -DOMPI_SKIP_MPICXX -std=c++0x -O3 -fPIC

# Include directories
INCLUDES := -I/usr/local/cuda/include -I/usr/include/jsoncpp -I../src/engine

# Linker flags and libraries
LDFLAGS := -L/usr/local/cuda/lib
LDLIBS := -lcudart

# Source files and object files
SRC := tensorhubcuda.c
OBJ := $(SRC:.c=.o)

# Target name
TARGET := libtensorhubcuda.a

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJ)
	ar -cvr $@ $^

# Rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean the generated files
.PHONY: clean
clean:
	rm -f $(OBJ) $(TARGET)
