#include "Air.h"

#include "raytracer/Vector.h"
#include "raytracer/Color.h"

Color Air::getColor(Vector point) {
   return Color(0.0, 0.0, 0.0);
}

double Air::getRefractiveIndex() {
   return AIR_REFRACTIVE_INDEX;
}
