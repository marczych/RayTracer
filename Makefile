CC = g++
CFLAGS = -O2 -Wall

RayTracer: main.o Image.o
	$(CC) $(CFLAGS) main.o Image.o -o RayTracer

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o

Image.o: Image.cpp Image.h
	$(CC) $(CFLAGS) Image.cpp -c -o Image.o

clean:
	rm -rf *.o
