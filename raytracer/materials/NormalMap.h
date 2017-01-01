#ifndef __NORMAL_MAP_H__
#define __NORMAL_MAP_H__

#include <iostream>

#include "PerlinNoise.h"
#include "raytracer/Vector.h"

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
