#ifndef __CRISS_CROSS_H__
#define __CRISS_CROSS_H__

#include "Material.h"
#include "PerlinNoise.h"
#include "raytracer/Color.h"

class CrissCross : public Material {
private:
   PerlinNoise perlin;

   Color color1;
   Color color2;
   Color color3;
   double scale;
   double shininess;
   double reflectivity;

public:
   CrissCross(std::istream&);

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
