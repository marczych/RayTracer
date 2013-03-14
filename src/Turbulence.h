#ifndef __TURBULENCE_H__
#define __TURBULENCE_H__

#include <iostream>
#include "Material.h"
#include "Color.h"
#include "PerlinNoise.h"

class Turbulence : public Material {
private:
   PerlinNoise perlin;

   Color color1;
   Color color2;
   double scale;
   double shininess;
   double reflectivity;

public:
   Turbulence(std::istream&);

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
