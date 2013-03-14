#include "CrissCross.h"

#include "Vector.h"
#include "Color.h"

CrissCross::CrissCross(std::istream& in) {
   in >> color1.r >> color1.g >> color1.b;
   in >> color2.r >> color2.g >> color2.b;
   in >> color3.r >> color3.g >> color3.b;
   in >> scale;
   in >> shininess;
   in >> reflectivity;
}

Color CrissCross::getColor(Vector point) {
   double x = point.x * scale * 0.5;
   double y = point.y * scale * 0.5;
   double z = point.z * scale * 0.5;
   double noiseCoefA = 0;
   double noiseCoefB = 0;
   double noiseCoefC = 0;

   for (int level = 1; level < 10; level++) {
      noiseCoefA += (1.0f / level) * fabsf(perlin.noise(
         level * 0.35 * x,
         level * 0.05 * y,
         level * z
      ));

      noiseCoefB += (1.0f / level) * fabsf(perlin.noise(
         level * x,
         level * 0.35 * y,
         level * 0.05 * z
      ));

      noiseCoefC += (1.0f / level) * fabsf(perlin.noise(
         level * 0.05 * x,
         level * y,
         level * 0.35 * z
      ));
   }
   noiseCoefA = 0.5f * sinf((x + z) * 0.05f + noiseCoefA) + 0.5f;
   noiseCoefB = 0.5f * sinf((y + x) * 0.05f + noiseCoefB) + 0.5f;
   noiseCoefC = 0.5f * sinf((z + y) * 0.05f + noiseCoefC) + 0.5f;

   return (color1 * noiseCoefA + color2 * (1.0f - noiseCoefA)) * 0.25 +
          (color2 * noiseCoefB + color3 * (1.0f - noiseCoefB)) * 0.25 +
          (color3 * noiseCoefC + color1 * (1.0f - noiseCoefC)) * 0.25;
}

double CrissCross::getShininess() {
   return shininess;
}

double CrissCross::getReflectivity() {
   return reflectivity;
}
