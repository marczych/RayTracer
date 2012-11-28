CC = g++
CFLAGS = -O2 -Wall

RayTracer: main.o Image.o Ray.o Vector.o Sphere.o Intersection.o Object.o Color.o Light.o
	$(CC) $(CFLAGS) main.o Image.o Ray.o Vector.o Sphere.o Intersection.o Object.o Color.o Light.o -o RayTracer

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o

Image.o: Image.cpp Image.h Color.h
	$(CC) $(CFLAGS) Image.cpp -c -o Image.o

Ray.o: Ray.cpp Ray.h Vector.h
	$(CC) $(CFLAGS) Ray.cpp -c -o Ray.o

Vector.o: Vector.cpp Vector.h
	$(CC) $(CFLAGS) Vector.cpp -c -o Vector.o

Sphere.o: Sphere.cpp Sphere.h Ray.h Vector.h Object.h Intersection.h
	$(CC) $(CFLAGS) Sphere.cpp -c -o Sphere.o

Intersection.o: Intersection.cpp Intersection.h Vector.h
	$(CC) $(CFLAGS) Intersection.cpp -c -o Intersection.o

Object.o: Object.cpp Object.h
	$(CC) $(CFLAGS) Object.cpp -c -o Object.o

Color.o: Color.cpp Color.h
	$(CC) $(CFLAGS) Color.cpp -c -o Color.o

Light.o: Light.cpp Light.h Vector.h
	$(CC) $(CFLAGS) Light.cpp -c -o Light.o

clean:
	rm -rf *.o
