#ifndef __RAY_H__
#define __RAY_H__

#include <stdlib.h>
#include "Vector.h"

class Material;

class Ray {
public:
   Vector origin;
   Vector direction;
   Vector fracDir;
   int reflectionsRemaining;
   Material* material;

   Ray() : origin(Vector()), direction(Vector()), reflectionsRemaining(-1),
    material(NULL) {
      calcFracDirection();
   }

   Ray(Vector origin_, Vector direction_, int reflections,
    Material* material_) : origin(origin_), reflectionsRemaining(reflections),
    material(material_) {
      direction = direction_.normalize();

      /* Move intersection slightly forward to avoid intersecting with itself. */
      origin += (direction / 1000);

      calcFracDirection();
   }

private:
   void calcFracDirection();
};

#endif
