#ifndef __NORMAL_MAP_H__
#define __NORMAL_MAP_H__

#include <iostream>
#include "Vector.h"
#include "PerlinNoise.h"

class NormalMap {
private:
   PerlinNoise perlin;
   double scale;
   double amount;

public:
   NormalMap(std::istream&);

   Vector modifyNormal(const Vector&, const Vector&);
};

#endif
