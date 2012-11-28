#ifndef __INTERSECTION_H__
#define __INTERSECTION_H__

#include <stdlib.h>
#include "Vector.h"
#include "Object.h"

class Intersection {
public:
   bool didIntersect;
   Vector intersection;
   Object* object;

   Intersection(Vector intersection_, Object* object_) :
    didIntersect(true), intersection(intersection_), object(object_) {}
   Intersection(bool didIntersect_) :
    didIntersect(didIntersect_), intersection(Vector(0, 0, 0)), object(NULL) {}
};

#endif
