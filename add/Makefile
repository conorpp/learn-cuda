CC = nvcc
src = $(wildcard *.cu)
obj = $(src:.cu=.o)

LDFLAGS = 
CFLAGS = -O3

name = $(shell basename $(CURDIR))

$(name): $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm -f $(obj) $(name)
