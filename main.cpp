#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include "Image.h"
#include "Sphere.h"
#include "Intersection.h"

using namespace std;

class RayTracer {
public:
   int width;
   int height;

   vector<Sphere> spheres;

   RayTracer(int width_, int height_) : width(width_), height(height_) {
      spheres.push_back(Sphere(Vector(250, 250, 250), 100));
   }

   void traceRays(string);
   Color castRay(int, int);
};

void RayTracer::traceRays(string fileName) {
   Image image(width, height);

   for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
         image.pixel(x, y, castRay(x, y));
      }
   }

   image.WriteTga(fileName.c_str(), true);
}

Color RayTracer::castRay(int x, int y) {
   Ray ray(Vector(x, y, -100), Vector(x, y, -99));

   for (vector<Sphere>::iterator itr = spheres.begin(); itr < spheres.end(); itr++) {
      Intersection intersection = itr->intersect(ray);

      if (intersection.didIntersect) {
         return Color(1.0, 1.0, 1.0);
      }
   }

   return Color(0,0, 0.0, 0.0);
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
