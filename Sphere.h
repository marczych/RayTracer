#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"

class Sphere : Object {
public:
   Vector center;
   double radius;

   Sphere(Vector center_, double radius_) : center(center_), radius(radius_) {}

   virtual Intersection intersect(Ray);
};

#endif
