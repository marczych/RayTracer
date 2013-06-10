#ifndef __RAY_TRACER_H__
#define __RAY_TRACER_H__

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "Vector.h"
#include "Camera.h"

#include <iostream>
#include <fstream>
#include <algorithm>

class Ray;
class Color;
class Intersection;
class Object;
class Light;
class Material;
class NormalMap;

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
   Material* startingMaterial;

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
   void readModel(std::string, int size, Vector translate, Material* material);

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
   double getReflectance(const Vector&, const Vector&, double, double);
   Vector refractVector(const Vector&, const Vector&, double, double);
   Vector reflectVector(Vector, Vector);
   Material* readMaterial(std::istream&);
   NormalMap* readNormalMap(std::istream&);
   void addMaterial(std::istream&);
};

#endif
