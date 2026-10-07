# Builds one standalone executable per test_*.cpp / test_*.cu file in this
# directory, linked against the static libgravity (built on demand).
#
#   make                  build all tests
#   make run              build and run all tests; exit status is non-zero on failure
#   make run-test_saxpy   build and run a single test
#   make test_saxpy       build a single test
#   make clean
#
# BUILD, GPU_ARCH and CUDA_PATH are passed through to the library build.

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc
CXX       ?= g++
GPU_ARCH  ?= native
BUILD     ?= release

LIB_SRC_DIR := ../src
BUILD_DIR   ?= ../build
LIB         := $(BUILD_DIR)/lib/libgravity.a
BIN_DIR     := $(BUILD_DIR)/tests

INCLUDES  := -I$(LIB_SRC_DIR)/include -I.
NVCCFLAGS := -std=c++17 -arch=$(GPU_ARCH) -Xcompiler -Wall,-Wextra $(INCLUDES)
CXXFLAGS  := -std=c++17 -Wall -Wextra $(INCLUDES) -I$(CUDA_PATH)/include
LDLIBS    := -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -lcudart

ifeq ($(BUILD),debug)
  NVCCFLAGS += -g -G -O0
  CXXFLAGS  += -g -O0
else
  NVCCFLAGS += -O2 -lineinfo
  CXXFLAGS  += -O2
endif

TEST_NAMES := $(basename $(wildcard test_*.cpp test_*.cu))
TESTS      := $(addprefix $(BIN_DIR)/,$(TEST_NAMES))

.PHONY: all run clean FORCE $(TEST_NAMES) $(addprefix run-,$(TEST_NAMES))

all: $(TESTS)

# Always ask the library Makefile to bring the archive up to date; the tests
# only relink when the archive actually changes.
$(LIB): FORCE
	$(MAKE) -C $(LIB_SRC_DIR) static BUILD_DIR=$(abspath $(BUILD_DIR))

$(BIN_DIR)/%: %.cpp $(LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $@.d $< -o $@ $(LIB) $(LDLIBS)

$(BIN_DIR)/%: %.cu $(LIB) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $@.d $< -o $@ $(LIB)

$(BIN_DIR):
	mkdir -p $@

# `make test_foo` builds one test.
$(TEST_NAMES): %: $(BIN_DIR)/%

# `make run-test_foo` builds and runs one test.
$(addprefix run-,$(TEST_NAMES)): run-%: $(BIN_DIR)/%
	$<

run: $(TESTS)
	@pass=0; fail=0; failed=""; \
	for t in $(TESTS); do \
	  echo "== $$(basename $$t)"; \
	  if $$t; then pass=$$((pass + 1)); else fail=$$((fail + 1)); failed="$$failed $$(basename $$t)"; fi; \
	done; \
	echo "----"; \
	echo "$$pass passed, $$fail failed"; \
	if [ $$fail -ne 0 ]; then echo "failed:$$failed"; exit 1; fi

clean:
	rm -rf $(BIN_DIR)

FORCE:

-include $(TESTS:=.d)
