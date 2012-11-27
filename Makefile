CC = g++
CFLAGS = -O2 -Wall

RayTracer: main.o Image.o Ray.o Vector.o Sphere.o Intersection.o
	$(CC) $(CFLAGS) main.o Image.o Ray.o Vector.o Sphere.o Intersection.o -o RayTracer

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o

Image.o: Image.cpp Image.h
	$(CC) $(CFLAGS) Image.cpp -c -o Image.o

Ray.o: Ray.cpp Ray.h Vector.h
	$(CC) $(CFLAGS) Ray.cpp -c -o Ray.o

Vector.o: Vector.cpp Vector.h
	$(CC) $(CFLAGS) Vector.cpp -c -o Vector.o

Sphere.o: Sphere.cpp Sphere.h Ray.h Vector.h
	$(CC) $(CFLAGS) Sphere.cpp -c -o Sphere.o

Intersection.o: Intersection.cpp Intersection.h Vector.h
	$(CC) $(CFLAGS) Intersection.cpp -c -o Intersection.o

clean:
	rm -rf *.o
