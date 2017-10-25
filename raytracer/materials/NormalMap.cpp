#include "NormalMap.h"

NormalMap::NormalMap(std::istream& in) {
   in >> scale;
   in >> amount;
}

Vector NormalMap::modifyNormal(const Vector& normal, const Vector& point) {
   Vector noise;
   double x = point.x / scale;
   double y = point.y / scale;
   double z = point.z / scale;

   noise.x = (float)(perlin.noise(x, y, z));
   noise.y = (float)(perlin.noise(y, z, x));
   noise.z = (float)(perlin.noise(z, x, y));

   return (normal + noise * amount).normalize();
}
