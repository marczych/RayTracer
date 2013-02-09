#ifndef __RAY_TRACER_H__
#define __RAY_TRACER_H__

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>
#include <vector>
#include <iostream>
#include "Vector.h"
#include "Camera.h"
#include "Sphere.h"
#include "Light.h"

class Ray;
class Color;
class Intersection;

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
   unsigned long numSpheres;
   unsigned long numLights;

   std::vector<Sphere> spheres;
   std::vector<Light> lights;

   RayTracer(int, int, int, int, int);

   ~RayTracer();

   void addObject(Sphere* object) {
      spheres.push_back(*object);
   }

   void addLight(Light* light) {
      lights.push_back(*light);
   }

   void traceRays(uchar4*, Sphere*, Light*);
   __device__ Color castRayForPixel(int, int, Sphere*, Light*);
   __device__ Color castRayAtPoint(Vector, Sphere*, Light*);
   __device__ Color castRay(Ray, Sphere*, Light*);
   __device__ Intersection getClosestIntersection(Ray, Sphere*);
   __device__ bool isInShadow(Ray, Sphere*, double distance);
   __device__ Color performLighting(Intersection, Light*, Sphere*);
   __device__ Color getAmbientLighting(Intersection);
   __device__ Color getDiffuseAndSpecularLighting(Intersection, Light*, Sphere*);
   __device__ Color getSpecularLighting(Intersection, Light*);
   __device__ Color getReflectiveLighting(Intersection);
   __device__ Vector reflectVector(Vector, Vector);
   void readScene(std::istream&);
};

__global__ void cudaTraceRays(Sphere* spheres, Light* lights, uchar4* image, RayTracer* rayTracer);

#endif
