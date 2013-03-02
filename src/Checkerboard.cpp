#include "Checkerboard.h"

#include "Vector.h"

// Offset points to move the origin which makes an ugly seam.
#define POINT_OFFSET 3893343

Color Checkerboard::getColor(Vector point) {
   bool x = (int)((point.x + POINT_OFFSET) / scale) % 2 == 0;
   bool y = (int)((point.y + POINT_OFFSET) / scale) % 2 == 0;
   bool z = (int)((point.z + POINT_OFFSET) / scale) % 2 == 0;

   if (x xor y xor z) {
      return color1;
   } else {
      return color2;
   }
}

double Checkerboard::getShininess() {
   return shininess;
}

double Checkerboard::getReflectivity() {
   return reflectivity;
}
