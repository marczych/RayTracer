#include "Glass.h"

#include "Vector.h"
#include "Color.h"

Glass::Glass(std::istream& in) {
   in >> refractiveIndex;
   in >> shininess;
}

Color Glass::getColor(Vector point) {
   return Color(0.0, 0.0, 0.0);
}

double Glass::getRefractiveIndex() {
   return refractiveIndex;
}

double Glass::getShininess() {
   return shininess;
}
