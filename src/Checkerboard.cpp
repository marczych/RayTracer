#include "Checkerboard.h"

#include "Vector.h"
#include "Color.h"

Color Checkerboard::getColor(Vector point) {
   bool x = (int)(point.x * 0.1) % 2 == 0;
   bool y = (int)(point.y * 0.1) % 2 == 0;
   bool z = (int)(point.z * 0.1) % 2 == 0;

   if (x xor y xor z) {
      return Color(0.4980f, 0.2980f, 0.8f);
   } else {
      return Color(0.8313f, 0.6352f, 0.3921f);
   }
}
