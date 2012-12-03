#ifndef __RAY_TRACER_H__
#define __RAY_TRACER_H__

#include <string>
#include <vector>
#include <iostream>

class Ray;
class Color;
class Intersection;
class Vector;
class Object;
class Light;

class RayTracer {
public:
   int width;
   int height;
   int maxReflections;
   int superSamples; // Square root of number of samples to use for each pixel.

   std::vector<Object*> objects;
   std::vector<Light*> lights;

   RayTracer(int width_, int height_, int maxReflections_, int superSamples_) :
    width(width_), height(height_), maxReflections(maxReflections_),
    superSamples(superSamples_) {}

   ~RayTracer();

   void addObject(Object* object) {
      objects.push_back(object);
   }

   void addLight(Light* light) {
      lights.push_back(light);
   }

   void traceRays(std::string);
   Color castRayForPixel(int, int);
   Color castRay(Ray);
   Intersection getClosestIntersection(Ray);
   Color performLighting(Intersection);
   Color getAmbientLighting(Intersection);
   Color getDiffuseAndSpecularLighting(Intersection);
   Color getSpecularLighting(Intersection, Light*);
   Color getReflectiveLighting(Intersection);
   Vector reflectVector(Vector, Vector);
   void readScene(std::istream&);
};

#endif
