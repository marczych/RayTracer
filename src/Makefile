CC = nvcc
CFLAGS = -O2 -arch=compute_20 -code=sm_20 -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU

ALL = RayTracer.cu callbacksPBO.cpp kernelPBO.cu simpleGLmain.cpp simplePBO.cpp

RayTracer: $(ALL)
	$(CC) $(CFLAGS) $(ALL) -o RayTracer

clean:
	rm -rf RayTracer
