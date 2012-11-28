#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include "Image.h"
#include "Object.h"
#include "Sphere.h"
#include "Intersection.h"

using namespace std;

class RayTracer {
public:
   int width;
   int height;

   vector<Object*> objects;

   RayTracer(int width_, int height_) : width(width_), height(height_) {
      objects.push_back(new Sphere(Vector(-100, 0, 0), 150, Color(1.0, 0.0, 0.0)));
      objects.push_back(new Sphere(Vector(100, 0, 0), 150, Color(0.0, 1.0, 0.0)));
   }

   ~RayTracer();

   void traceRays(string);
   Color castRay(int, int);
};

RayTracer::~RayTracer() {
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      delete *itr;
   }
}

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
   int rayX = x - width / 2;
   int rayY = y - height / 2;
   Ray ray(Vector(rayX, rayY, -100), Vector(rayX, rayY, -99));

   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect) {
         return intersection.color;
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
