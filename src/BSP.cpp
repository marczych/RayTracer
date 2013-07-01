#include "BSP.h"
#include <limits>
#include "Object.h"
#include "Intersection.h"

using namespace std;

void BSP::build(bool increment) {
   // We've hit our limit. This is a leaf node.
   if (objects.size() <= MIN_OBJECT_COUNT) {
      return;
   }
   for (int i = 0; i < depth; i++) {
     cout << "\t";
   }
   cout << objects.size() << endl;

   // Make sure all objects are properly wrapped
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
     Boundaries curr = (*itr)->getBounds();
     bounds.min = Vector(min(bounds.min.x, curr.min.x),
                         min(bounds.min.y, curr.min.y),
                         min(bounds.min.z, curr.min.z));
     bounds.max = Vector(max(bounds.max.x, curr.max.x),
                         max(bounds.max.y, curr.max.y),
                         max(bounds.max.z, curr.max.z));
   }

   // Where to split the bounds
   double splitValue = bounds.splitValue(axis);

   vector<Object*> leftObjects;
   vector<Object*> rightObjects;

   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Object* obj = *itr;
      Boundaries curr = obj->getBounds();
      double min, max;

      switch(axis) {
         case 'x':
            min = curr.min.x;
            max = curr.max.x;
            break;
         case 'y':
            min = curr.min.y;
            max = curr.max.y;
            break;
         case 'z':
            min = curr.min.z;
            max = curr.max.z;
            break;
      }

      if (min < splitValue) {
         leftObjects.push_back(obj);
      }

      if (max > splitValue) {
         rightObjects.push_back(obj);
      }
   }

   int newAxis = toggleAxis();

   if (leftObjects.size() != objects.size() &&
       rightObjects.size() != objects.size()) {
      // Since this split separated geometry a little bit, make children to
      // split up geometry further.
      left = new BSP(depth + 1, newAxis, leftObjects);
      right = new BSP(depth + 1, newAxis, rightObjects);
   } else if (axisRetries == 2) {
      // Splitting objects on this axis didn't achieve anything.
      left = right = NULL;
   } else {
      axis = toggleAxis();
      axisRetries++;
      build(true);
   }
}

char BSP::toggleAxis() {
   return axis == 'x' ? 'y' : (axis == 'y' ? 'z' : 'x');
}

/**
 * Given a ray and an axis aligned bounding box, determine whether the ray intersects.
 */
bool BSP::intersectAABB(const Ray& ray, Boundaries bounds, double* dist) {
   double txmin = (bounds.min.x - ray.origin.x) / ray.direction.x;
   double txmax = (bounds.max.x - ray.origin.x) / ray.direction.x;
   if (txmin > txmax)
      swap(txmin, txmax);

   double tymin = (bounds.min.y - ray.origin.y) / ray.direction.y;
   double tymax = (bounds.max.y - ray.origin.y) / ray.direction.y;
   if (tymin > tymax)
      swap(tymin, tymax);

   if ((txmin > tymax) || (tymin > txmax))
      return false;

   if (tymin > txmin)
      txmin = tymin;

   if (tymax < txmax)
      txmax = tymax;

   double tzmin = (bounds.min.z - ray.origin.z) / ray.direction.z;
   double tzmax = (bounds.max.z - ray.origin.z) / ray.direction.z;
   if (tzmin > tzmax)
      swap(tzmin, tzmax);

   if ((txmin > tzmax) || (tzmin > txmax))
      return false;

   if (tzmin > txmin)
      txmin = tzmin;

   if (tzmax < txmax)
      txmax = tzmax;

   if ((txmin > bounds.max.x) || (txmax < bounds.min.x))
      return false;

   // Return distance to intersection for tie-breakers
   Vector distV = Vector(txmin, tymin, tzmin) - ray.origin;
   double newDist = sqrt(distV.x * distV.x +
                         distV.y * distV.y +
                         distV.z * distV.z);
   *dist = newDist;
   return true;
}

Intersection BSP::getClosestIntersection(const Ray& ray) {
   double distL, distR;
   bool intersectL = false, intersectR = false;

   if (left && right) {
      // There are children! See if they block the ray
      intersectL = intersectAABB(ray, left->bounds, &distL);
      intersectR = intersectAABB(ray, right->bounds, &distR);

      // If both hit, follow nearest match
      if (intersectL && intersectR) {
         if (distL < distR) {
            return (*left).getClosestIntersection(ray);
         }
         return (*right).getClosestIntersection(ray);
      }

      if (intersectL) {
         return (*left).getClosestIntersection(ray);
      }

      if (intersectR) {
         return (*right).getClosestIntersection(ray);
      }
   }

   // No children so just go through current objects like normal.
   Intersection closestIntersection;
   closestIntersection.distance = numeric_limits<double>::max();

   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect && intersection.distance <
       closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection;
}
