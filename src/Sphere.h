#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Vector.h"
#include "Color.h"
#include "Ray.h"
#include "Intersection.h"

class Sphere {
public:
   Vector center;
   double radius;
   Color color;
   double shininess;
   double reflectivity;

   __device__ Sphere(Vector center_, double radius_, Color color_, double shininess_,
    double reflectivity_) : center(center_), radius(radius_), color(color_),
    shininess(shininess_), reflectivity(reflectivity_) {}

   __device__ Intersection intersect(Ray);
   __device__ double getShininess();
   __device__ double getReflectivity();
};

#endif
