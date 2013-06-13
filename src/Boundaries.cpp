#include "Boundaries.h"

using namespace std;


// For when I optimize shit
double Boundaries::splitValue(char axis) {
   switch(axis) {
      case 'x': return (min.x + max.x) / 2;
      case 'y': return (min.y + max.y) / 2;
      case 'z': return (min.z + max.z) / 2;
   }

   return 0.0;
}

void Boundaries::split(Boundaries * left, Boundaries * right, char axis) {
   double split = splitValue(axis);

   switch(axis) {
      case 'x':
         left->min = min;
         left->max = Vector(split, max.y, max.z);
         right->min = Vector(split, min.y, min.z);
         right->max = max;
         break;
      case 'y':
         left->min = min;
         left->max = Vector(max.x, split, max.z);
         right->min = Vector(min.x, split, min.z);
         right->max = max;
         break;
      case 'z':
         left->min = min;
         left->max = Vector(max.x, max.y, split);
         right->min = Vector(min.x, min.y, split);
         right->max = max;
         break;
   }
}
