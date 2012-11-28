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
   Vector normal;
   Color color;
   Object* object;

   Intersection(Vector intersection_, Vector normal_, Color color_, Object* object_) :
    didIntersect(true), intersection(intersection_), normal(normal_), color(color_), object(object_) {}
   Intersection(bool didIntersect_) :
    didIntersect(didIntersect_), intersection(Vector()), normal(Vector()), color(Color()), object(NULL) {}
};

#endif
