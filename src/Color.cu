#include "Color.h"

__device__ Color Color::operator+ (Color const &c) const {
   Color other;

   other.r = NTZ(c.r) + NTZ(r);
   other.g = NTZ(c.g) + NTZ(g);
   other.b = NTZ(c.b) + NTZ(b);

   return other;
}

__device__ Color Color::operator* (double amount) const {
   Color other;

   other.r = r * amount;
   other.g = g * amount;
   other.b = b * amount;

   return other;
}
