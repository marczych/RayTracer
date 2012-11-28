#ifndef __RAY_H__
#define __RAY_H__

#include "Vector.h"

class Ray {
public:
   Vector origin;
   Vector direction;

   Ray(Vector origin_, Vector direction_) : origin(origin_) {
      direction = direction_.normalize();
   }
};

#endif
