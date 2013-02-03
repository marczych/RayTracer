#ifndef __INTERSECTION_H__
#define __INTERSECTION_H__

#include <stdlib.h>
#include "Vector.h"
#include "Color.h"
#include "Ray.h"

class Sphere;

class Intersection {
public:
   Ray ray;
   bool didIntersect;
   Vector intersection;
   double distance;
   Vector normal;
   Color color;
   Sphere* object;

   __device__ Intersection(Ray ray_, Vector intersection_, double distance_, Vector normal_,
    Color color_, Sphere* object_) : ray(ray_), didIntersect(true),
    intersection(intersection_), distance(distance_), normal(normal_), color(color_),
    object(object_) {}

   __device__ Intersection(bool didIntersect_) : ray(Ray()), didIntersect(didIntersect_),
    intersection(Vector()), distance(0.0), normal(Vector()), color(Color()),
    object(NULL) {}
};

#endif
