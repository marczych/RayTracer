#ifndef __RAY_H__
#define __RAY_H__

#include "Vector.h"
#include "Material.h"
#include <cstddef>

class Ray {
public:
   Vector origin;
   Vector direction;
   int reflectionsRemaining;
   Material* material;

   Ray() : origin(Vector()), direction(Vector()), reflectionsRemaining(-1),
    material(NULL) {}

   Ray(Vector origin_, Vector direction_, int reflections,
    Material* material_) : origin(origin_), reflectionsRemaining(reflections),
    material(material_) {
      direction = direction_.normalize();

      /* Move intersection slightly forward to avoid intersecting with itself. */
      origin += (direction / 1000);
   }
};

#endif
