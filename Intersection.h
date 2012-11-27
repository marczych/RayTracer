#ifndef __INTERSECTION_H__
#define __INTERSECTION_H__

#include "Vector.h"

class Intersection {
public:
   bool didIntersect;
   Vector intersection;

   Intersection(Vector intersection_) : didIntersect(true), intersection(intersection_) {}
   Intersection(bool didIntersect_) : didIntersect(didIntersect_), intersection(Vector(0, 0, 0)) {}
};

#endif
