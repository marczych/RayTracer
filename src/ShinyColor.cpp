#include "ShinyColor.h"

#include "Vector.h"
#include "Color.h"

ShinyColor::ShinyColor(std::istream& in) {
   in >> color.r >> color.g >> color.b;
   in >> shininess;
   in >> reflectivity;
}

Color ShinyColor::getColor(Vector point) {
   return color;
}

double ShinyColor::getShininess() {
   return shininess;
}

double ShinyColor::getReflectivity() {
   return reflectivity;
}
