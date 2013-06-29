#include "Boundaries.h"

// Returns the center value for the given axis.
double Boundaries::splitValue(char axis) {
   switch(axis) {
      case 'x': return (min.x + max.x) / 2;
      case 'y': return (min.y + max.y) / 2;
      case 'z': return (min.z + max.z) / 2;
      default: return 0.0f;
   }
}
