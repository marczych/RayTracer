#ifndef __BOUNDARIES_H__
#define __BOUNDARIES_H__

#include "Vector.h"

class Boundaries {
public:
   Vector min, max;

   Boundaries() : min(Vector(0, 0, 0)), max(Vector(0, 0, 0)) {}
   Boundaries(Vector min_, Vector max_) : min(min_), max(max_) {}
   Boundaries(const Boundaries& other) : min(other.min), max(other.max) {}

   double splitValue(char axis);
};

#endif
