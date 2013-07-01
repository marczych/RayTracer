#include "Boundaries.h"
#include <math.h>
#include <algorithm>

using namespace std;

// Returns the center value for the given axis.
double Boundaries::splitValue(char axis) {
   switch(axis) {
      case 'x': return (min.x + max.x) / 2;
      case 'y': return (min.y + max.y) / 2;
      case 'z': return (min.z + max.z) / 2;
      default: return 0.0f;
   }
}

/**
 * Ray axis aligned bounding box intersection.
 */
bool Boundaries::intersect(const Ray& ray, double* dist) {
   double txmin = (min.x - ray.origin.x) / ray.direction.x;
   double txmax = (max.x - ray.origin.x) / ray.direction.x;
   if (txmin > txmax)
      swap(txmin, txmax);

   double tymin = (min.y - ray.origin.y) / ray.direction.y;
   double tymax = (max.y - ray.origin.y) / ray.direction.y;
   if (tymin > tymax)
      swap(tymin, tymax);

   if ((txmin > tymax) || (tymin > txmax))
      return false;

   if (tymin > txmin)
      txmin = tymin;

   if (tymax < txmax)
      txmax = tymax;

   double tzmin = (min.z - ray.origin.z) / ray.direction.z;
   double tzmax = (max.z - ray.origin.z) / ray.direction.z;
   if (tzmin > tzmax)
      swap(tzmin, tzmax);

   if ((txmin > tzmax) || (tzmin > txmax))
      return false;

   if (tzmin > txmin)
      txmin = tzmin;

   if (tzmax < txmax)
      txmax = tzmax;

   if ((txmin > max.x) || (txmax < min.x))
      return false;

   // Return distance to intersection for tie-breakers
   Vector distV = Vector(txmin, tymin, tzmin) - ray.origin;
   double newDist = sqrt(distV.x * distV.x +
                         distV.y * distV.y +
                         distV.z * distV.z);
   *dist = newDist;
   return true;
}
