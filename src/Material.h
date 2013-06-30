#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#define NOT_SHINY -1
#define NOT_REFLECTIVE -1
#define NOT_REFRACTIVE -1
#define AIR_REFRACTIVE_INDEX 1

#include <stdlib.h>
#include <iostream>
#include "Color.h"
#include "PerlinNoise.h"

class Vector;
class Color;
class NormalMap;

class Material {
private:
   NormalMap* normalMap;

public:
   void setNormalMap(NormalMap* normalMap_) { normalMap = normalMap_; }

   virtual Color getColor(Vector) = 0;
   virtual double getShininess();
   virtual double getReflectivity();
   virtual double getRefractiveIndex();

   Vector modifyNormal(const Vector&, const Vector&);
};

#endif
