#include "Color.h"

Color Color::operator+ (Color const &c) const {
   Color other;

   other.r = c.r + r;
   other.g = c.g + g;
   other.b = c.b + b;

   return other;
}

Color Color::operator* (double amount) const {
   Color other;

   other.r = r * amount;
   other.g = g * amount;
   other.b = b * amount;

   return other;
}
