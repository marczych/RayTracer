#include "FlatColor.h"

#include "raytracer/Vector.h"
#include "raytracer/Color.h"

FlatColor::FlatColor(std::istream& in) {
   in >> color.r >> color.g >> color.b;
}

Color FlatColor::getColor(Vector point) {
   return color;
}
