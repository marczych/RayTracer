#ifndef __INTERSECTION_H__
#define __INTERSECTION_H__

#include <stdlib.h>
#include "Vector.h"
#include "Object.h"
#include "Color.h"
#include "Ray.h"

class Intersection {
public:
   Ray ray;
   bool didIntersect;
   Vector intersection;
   double distance;
   Vector normal;
   Color color;
   Object* object;

   Intersection(Ray ray_, Vector intersection_, double distance_, Vector normal_,
    Color color_, Object* object_) : ray(ray_), didIntersect(true),
    intersection(intersection_), distance(distance_), normal(normal_), color(color_),
    object(object_) {}

   Intersection(bool didIntersect_) : ray(Ray()), didIntersect(didIntersect_),
    intersection(Vector()), distance(0.0), normal(Vector()), color(Color()),
    object(NULL) {}
};

#endif
