#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"

class Sphere {
public:
   Vector center;
   double radius;

   Sphere(Vector center_, double radius_) : center(center_), radius(radius_) {}

   Intersection intersect(Ray);
};

#endif
