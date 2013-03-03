#ifndef __RAY_H__
#define __RAY_H__

#include "Vector.h"
#include "Material.h"

class Ray {
public:
   Vector origin;
   Vector direction;
   int reflectionsRemaining;
   double refractiveIndex;

   Ray() : origin(Vector()), direction(Vector()), reflectionsRemaining(-1),
    refractiveIndex(AIR_REFRACTIVE_INDEX) {}

   Ray(Vector origin_, Vector direction_, int reflections,
    double refractiveIndex_) : origin(origin_), reflectionsRemaining(reflections),
    refractiveIndex(refractiveIndex_) {
      direction = direction_.normalize();

      /* Move intersection slightly forward to avoid intersecting with itself. */
      origin += (direction / 1000);
   }
};

#endif
