#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "Vector.h"

class Light {
public:
   Vector position;

   Light(Vector position_) : position(position_) {}
};

#endif
