#include "FlatColor.h"

#include "Vector.h"
#include "Color.h"

Color FlatColor::getColor(Vector point) {
   return color;
}

double FlatColor::getShininess() {
   return shininess;
}

double FlatColor::getReflectivity() {
   return reflectivity;
}
