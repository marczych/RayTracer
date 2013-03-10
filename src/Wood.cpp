#include "Wood.h"

#include "Vector.h"
#include "Color.h"

Wood::Wood(std::istream& in) {
   in >> color1.r >> color1.g >> color1.b;
   in >> color2.r >> color2.g >> color2.b;
   in >> scale;
   in >> shininess;
   in >> reflectivity;
}

Color Wood::getColor(Vector point) {
   double x = point.x * scale;
   double y = point.y * scale;
   double z = point.z * scale;

   double grain = perlin.noise(x, y, z) * 5;

   grain = grain - (int)grain;

   return color1 * grain + color2 * (1.0f - grain);
}

double Wood::getShininess() {
   return shininess;
}

double Wood::getReflectivity() {
   return reflectivity;
}
