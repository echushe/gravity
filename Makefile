# Convenience wrapper around src/Makefile and tests/Makefile.
# Variables given on the command line (BUILD, GPU_ARCH, CUDA_PATH, ...) are
# passed through to both.

.PHONY: all lib tests test clean

all: lib tests

lib:
	$(MAKE) -C src

tests: lib
	$(MAKE) -C tests

test: lib
	$(MAKE) -C tests run

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
