CC = nvcc
CFLAGS = -O2 -arch=compute_20 -code=sm_20

RayTracer: RayTracer.cu main.cu
	$(CC) $(CFLAGS) RayTracer.cu main.cu -o RayTracer

clean:
	rm -rf RayTracer
