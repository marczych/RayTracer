CC = g++
CFLAGS = -O2 -Wall

RayTracer: main.o Image.o Ray.o
	$(CC) $(CFLAGS) main.o Image.o Ray.o -o RayTracer

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o

Image.o: Image.cpp Image.h
	$(CC) $(CFLAGS) Image.cpp -c -o Image.o

Ray.o: Ray.cpp Ray.h Vector.h
	$(CC) $(CFLAGS) Ray.cpp -c -o Ray.o

Vector.o: Vector.cpp Vector.h
	$(CC) $(CFLAGS) Vector.cpp -c -o Vector.o

clean:
	rm -rf *.o
