#ifndef __INTERSECTION_H__
#define __INTERSECTION_H__

#include <stdlib.h>
#include "Vector.h"
#include "Object.h"
#include "Color.h"

class Intersection {
public:
   bool didIntersect;
   Vector intersection;
   double distance;
   Vector normal;
   Color color;
   Object* object;

   Intersection(Vector intersection_, double distance_, Vector normal_, Color color_, Object* object_) :
    didIntersect(true), intersection(intersection_), distance(distance_), normal(normal_), color(color_),
    object(object_) {}

   Intersection(bool didIntersect_) :
    didIntersect(didIntersect_), intersection(Vector()), distance(0.0), normal(Vector()), color(Color()),
    object(NULL) {}
};

#endif
