TARGETS := test

all: $(TARGETS)

$(TARGETS): %: %.o
	clang -o $@ $< -lcuda -lcudart -lcublas -L"/usr/local/cuda/lib64"

%.o: %.cu
	nvcc $< -o $@ -c -O3 -lineinfo --use_fast_math -arch=sm_35
