#ifndef __RAY_H__
#define __RAY_H__

#include "Vector.h"

class Ray {
public:
   Vector origin;
   Vector direction;
   int reflectionsRemaining;

   __device__ Ray() : origin(Vector()), direction(Vector()), reflectionsRemaining(-1) {}

   __device__ Ray(Vector origin_, Vector direction_, int reflections) :
    origin(origin_), reflectionsRemaining(reflections) {
      direction = direction_.normalize();

      /* Move intersection slightly forward to avoid intersecting with itself. */
      origin += (direction / 1000);
   }
};

#endif
