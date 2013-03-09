#include "Air.h"

#include "Vector.h"
#include "Color.h"

Color Air::getColor(Vector point) {
   return Color(0.0, 0.0, 0.0);
}

double Air::getRefractiveIndex() {
   return AIR_REFRACTIVE_INDEX;
}
