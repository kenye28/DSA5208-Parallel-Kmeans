CXX       ?= g++
MPICXX    ?= mpicxx
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS  ?= -fopenmp

.PHONY: all clean serial mpi mpi_openmp
all: bin/serial bin/mpi bin/mpi_openmp
serial: bin/serial
mpi: bin/mpi
mpi_openmp: bin/mpi_openmp

bin:
	mkdir -p bin

bin/serial: src/serial.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) -o $@ src/serial.cpp

bin/mpi: src/mpi.cpp src/common.hpp | bin
	$(MPICXX) $(CXXFLAGS) -o $@ src/mpi.cpp

bin/mpi_openmp: src/mpi_openmp.cpp src/common.hpp | bin
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ src/mpi_openmp.cpp

clean:
	rm -rf bin
