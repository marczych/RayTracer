CC = nvcc
CFLAGS = -O2

RayTracer: main.o RayTracer.o Image.o Ray.o Vector.o Sphere.o Intersection.o Object.o Color.o Light.o Camera.o
	$(CC) $(CFLAGS) main.o RayTracer.o Image.o Ray.o Vector.o Sphere.o Intersection.o Object.o Color.o Light.o Camera.o -o RayTracer

RayTracer.o: RayTracer.cu RayTracer.h Vector.h Image.h Object.h Sphere.h Intersection.h Light.h
	$(CC) $(CFLAGS) RayTracer.cu -c -o RayTracer.o

main.o: main.cu RayTracer.h
	$(CC) $(CFLAGS) main.cu -c -o main.o

Image.o: Image.cu Image.h Color.h
	$(CC) $(CFLAGS) Image.cu -c -o Image.o

Ray.o: Ray.cu Ray.h Vector.h
	$(CC) $(CFLAGS) Ray.cu -c -o Ray.o

Vector.o: Vector.cu Vector.h
	$(CC) $(CFLAGS) Vector.cu -c -o Vector.o

Sphere.o: Sphere.cu Sphere.h Ray.h Vector.h Object.h Intersection.h
	$(CC) $(CFLAGS) Sphere.cu -c -o Sphere.o

Intersection.o: Intersection.cu Intersection.h Vector.h
	$(CC) $(CFLAGS) Intersection.cu -c -o Intersection.o

Object.o: Object.cu Object.h
	$(CC) $(CFLAGS) Object.cu -c -o Object.o

Color.o: Color.cu Color.h
	$(CC) $(CFLAGS) Color.cu -c -o Color.o

Light.o: Light.cu Light.h Vector.h
	$(CC) $(CFLAGS) Light.cu -c -o Light.o

Camera.o: Camera.cu Camera.h Camera.h
	$(CC) $(CFLAGS) Camera.cu -c -o Camera.o

clean:
	rm -rf *.o
