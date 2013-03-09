#include "FlatColor.h"

#include "Vector.h"
#include "Color.h"

FlatColor::FlatColor(std::istream& in) {
   in >> color.r >> color.g >> color.b;
}

Color FlatColor::getColor(Vector point) {
   return color;
}
