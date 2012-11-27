#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "Image.h"

using namespace std;

class RayTracer {
public:
   int width;
   int height;

   RayTracer(int width_, int height_) : width(width_), height(height_) {}

   void traceRays(string);
};

void RayTracer::traceRays(string fileName) {
   Image image(width, height);

   image.pixel(250, 250, Color(0.5, 0, 0, 1));

   image.WriteTga(fileName.c_str(), true);
}

/**
 * RayTracer main.
 */
int main(void) {
   RayTracer rayTracer(500, 500);
   string fileName = "awesome.tga";

   rayTracer.traceRays(fileName);

   return 0;
}
