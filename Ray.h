#ifndef __RAY_H__
#define __RAY_H__

#include "Vector.h"

class Ray {
public:
   Vector origin;
   Vector direction;
   int reflectionsRemaining;

   Ray() : origin(Vector()), direction(Vector()), reflectionsRemaining(-1) {}

   Ray(Vector origin_, Vector direction_, int reflections) :
    origin(origin_), reflectionsRemaining(reflections) {
      direction = direction_.normalize();
   }
};

#endif
