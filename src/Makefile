# Builds libgravity (static and shared) from the .cu and .cpp files in this
# directory. Every .cu / .cpp file here is picked up automatically.
#
#   make                 build both libraries (release)
#   make static|shared   build only one of them
#   make BUILD=debug     debug build (-g -G -O0); run `make clean` when switching
#   make GPU_ARCH=sm_86  target a specific GPU instead of the local one
#   make clean

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc
CXX       ?= g++
GPU_ARCH  ?= native
BUILD     ?= release

BUILD_DIR ?= ../build
OBJ_DIR   := $(BUILD_DIR)/obj
LIB_DIR   := $(BUILD_DIR)/lib

LIB_NAME   := gravity
STATIC_LIB := $(LIB_DIR)/lib$(LIB_NAME).a
SHARED_LIB := $(LIB_DIR)/lib$(LIB_NAME).so

INCLUDES  := -Iinclude -I.
NVCCFLAGS := -std=c++17 -arch=$(GPU_ARCH) -Xcompiler -fPIC,-Wall,-Wextra $(INCLUDES)
CXXFLAGS  := -std=c++17 -fPIC -Wall -Wextra $(INCLUDES) -I$(CUDA_PATH)/include

ifeq ($(BUILD),debug)
  NVCCFLAGS += -g -G -O0
  CXXFLAGS  += -g -O0
else ifeq ($(BUILD),release)
  NVCCFLAGS += -O3 -lineinfo
  CXXFLAGS  += -O3
else
  $(error BUILD must be 'release' or 'debug', got '$(BUILD)')
endif

CU_SRCS  := $(wildcard *.cu)
CPP_SRCS := $(wildcard *.cpp)
OBJS     := $(CU_SRCS:%.cu=$(OBJ_DIR)/%.cu.o) $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.cpp.o)
DEPS     := $(OBJS:.o=.d)

.PHONY: all static shared clean

all: static shared
static: $(STATIC_LIB)
shared: $(SHARED_LIB)

$(STATIC_LIB): $(OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^

$(SHARED_LIB): $(OBJS) | $(LIB_DIR)
	$(NVCC) -shared -arch=$(GPU_ARCH) -o $@ $^

$(OBJ_DIR)/%.cu.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIR)/%.cpp.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(OBJ_DIR) $(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

-include $(DEPS)
