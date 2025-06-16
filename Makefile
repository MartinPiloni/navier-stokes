NVCC=nvcc
NVFLAGS=-std=c++11 -O3 -Xcompiler "-Wall -Wextra -Wno-unused-parameter -O3 -ffast-math -march=native -ftree-vectorize"
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.cu)
COMMON_OBJECTS=solver.o wtime.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.[ch]
	$(NVCC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
