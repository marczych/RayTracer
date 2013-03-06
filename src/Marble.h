#ifndef __MARBLE_H__
#define __MARBLE_H__

#include "Material.h"
#include "Color.h"
#include "PerlinNoise.h"

class Marble : public Material {
private:
   PerlinNoise perlin;

public:
   Color color1;
   Color color2;
   double shininess;
   double reflectivity;
   double refractiveIndex;
   double scale;

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
   virtual double getRefractiveIndex();
};

#endif
