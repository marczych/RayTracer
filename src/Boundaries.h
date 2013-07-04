#ifndef __BOUNDARIES_H__
#define __BOUNDARIES_H__

#include "Vector.h"
#include "Ray.h"

class Boundaries {
public:
   Vector min, max;

   Boundaries() : min(Vector(0, 0, 0)), max(Vector(0, 0, 0)) {}

   double splitValue(char axis);
   bool intersect(const Ray&, double* dist);
};

#endif
