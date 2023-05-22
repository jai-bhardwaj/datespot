SHELL := /bin/sh
VPATH :=

PREFIX ?= $(CURDIR)/tensorhub
BUILD_DIR ?= $(CURDIR)/build

SRC_DIRS := src/engine src/utils src/runtime tests

.PHONY: all engine utils runtime tests install run-tests clean

all: engine runtime utils tests

engine runtime utils tests:
	$(MAKE) -C $@

install: all
	mkdir -p $(PREFIX)
	cp -rfp $(BUILD_DIR)/$@/* $(PREFIX)

run-tests: tests
	$(MAKE) -C $@ run-tests

clean:
	for dir in $(SRC_DIRS); do \
		$(MAKE) -C $$dir clean; \
	done
