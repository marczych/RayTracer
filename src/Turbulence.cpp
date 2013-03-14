#include "Turbulence.h"

#include "Vector.h"
#include "Color.h"

Turbulence::Turbulence(std::istream& in) {
   in >> color1.r >> color1.g >> color1.b;
   in >> color2.r >> color2.g >> color2.b;
   in >> scale;
   in >> shininess;
   in >> reflectivity;
}

Color Turbulence::getColor(Vector point) {
   double x = point.x * scale;
   double y = point.y * scale;
   double z = point.z * scale;
   double noiseCoef = 0;

   for (int level = 1; level < 10; level ++) {
      noiseCoef += (1.0f / level) * fabsf(perlin.noise(
         level * 0.05 * x,
         level * 0.05 * y,
         level * 0.05 * z
      ));
   }

   return color1 * noiseCoef + color2 * (1.0f - noiseCoef);
}

double Turbulence::getShininess() {
   return shininess;
}

double Turbulence::getReflectivity() {
   return reflectivity;
}
