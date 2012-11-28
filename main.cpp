#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <limits>
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
      objects.push_back(new Sphere(Vector(0, 0, -1000), 200, Color(1.0, 0.0, 0.0)));
      objects.push_back(new Sphere(Vector(-200, 0, -200), 100, Color(0.0, 1.0, 0.0)));
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
   Ray ray(Vector(rayX, rayY, 100), Vector(0, 0, -1));
   Intersection closestIntersection(false);
   closestIntersection.distance = numeric_limits<double>::max();

   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect && intersection.distance < closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection.color;
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
