#ifndef __RAY_TRACER_H__
#define __RAY_TRACER_H__

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "Vector.h"
#include "Camera.h"

class Ray;
class Color;
class Intersection;
class Object;
class Light;
class Material;

class RayTracer {
public:
   int width;
   int height;
   int maxReflections;
   int superSamples; // Square root of number of samples to use for each pixel.
   Camera camera;
   double imageScale;
   int depthComplexity;
   double dispersion;
   unsigned long long raysCast;

   std::vector<Object*> objects;
   std::vector<Light*> lights;
   std::map<std::string, Material*> materials;

   RayTracer(int, int, int, int, int);

   ~RayTracer();

   void addObject(Object* object) {
      objects.push_back(object);
   }

   void addLight(Light* light) {
      lights.push_back(light);
   }

   void traceRays(std::string);
   void readScene(std::istream&);

private:
   Color castRayForPixel(int, int);
   Color castRayAtPoint(const Vector&);
   Color castRay(const Ray&);
   bool isInShadow(const Ray&, double);
   Intersection getClosestIntersection(const Ray&);
   Color performLighting(const Intersection&);
   Color getAmbientLighting(const Intersection&, const Color&);
   Color getDiffuseAndSpecularLighting(const Intersection&, const Color&);
   Color getSpecularLighting(const Intersection&, Light*);
   Color getReflectiveRefractiveLighting(const Intersection&);
   Vector reflectVector(Vector, Vector);
   Material* readMaterial(std::istream&);
   void addMaterial(std::istream&);
};

#endif
