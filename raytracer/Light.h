#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "Vector.h"

class Light {
public:
   Vector position;
   double intensity;

   Light(Vector position_, double intensity_) :
    position(position_), intensity(intensity_) {}
};

#endif
