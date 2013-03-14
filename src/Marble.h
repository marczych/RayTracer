#ifndef __MARBLE_H__
#define __MARBLE_H__

#include <iostream>
#include "Material.h"
#include "Color.h"
#include "PerlinNoise.h"

class Marble : public Material {
private:
   PerlinNoise perlin;

   Color color1;
   Color color2;
   double scale;
   double shininess;
   double reflectivity;

public:
   Marble(std::istream&);

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
