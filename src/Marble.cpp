#include "Marble.h"

#include "Vector.h"
#include "Color.h"

Marble::Marble(std::istream& in) {
   in >> color1.r >> color1.g >> color1.b;
   in >> color2.r >> color2.g >> color2.b;
   in >> scale;
   in >> shininess;
   in >> reflectivity;
}

Color Marble::getColor(Vector point) {
   double x = point.x * scale;
   double y = point.y * scale;
   double z = point.z * scale;
   double noiseCoef = 0;

   for (int level = 1; level < 10; level ++) {
      noiseCoef +=  (1.0f / level) * fabsf(perlin.noise(
         level * 0.05 * x,
         level * 0.15 * y,
         level * 0.05 * z
      ));
   }
   noiseCoef = 0.5f * sinf((x + y) * 0.05f + noiseCoef) + 0.5f;

   return color1 * noiseCoef + color2 * (1.0f - noiseCoef);
}

double Marble::getShininess() {
   return shininess;
}

double Marble::getReflectivity() {
   return reflectivity;
}
