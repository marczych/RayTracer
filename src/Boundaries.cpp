#include "Boundaries.h"
#include <algorithm>

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
 * Adapted from: http://gamedev.stackexchange.com/a/18459
 */
bool Boundaries::intersect(const Ray& ray, double* dist) {
   // TODO: Put this into the ray so it's only calculated once per ray.
   double fracx = 1.0f / ray.direction.x;
   double fracy = 1.0f / ray.direction.y;
   double fracz = 1.0f / ray.direction.z;

   // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
   // r.org is origin of ray
   double t1 = (min.x - ray.origin.x) * fracx;
   double t2 = (max.x - ray.origin.x) * fracx;
   double t3 = (min.y - ray.origin.y) * fracy;
   double t4 = (max.y - ray.origin.y) * fracy;
   double t5 = (min.z - ray.origin.z) * fracz;
   double t6 = (max.z - ray.origin.z) * fracz;

   double tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
   double tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

   // If tmax < 0, ray is intersecting AABB, but whole AABB is behind us.
   if (tmax < 0) {
      return false;
   }

   // If tmin > tmax, ray doesn't intersect AABB.
   if (tmin > tmax) {
      return false;
   }

   return true;
}
