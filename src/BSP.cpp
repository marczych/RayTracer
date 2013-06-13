#include "BSP.h"

using namespace std;

void BSP::build() {
   // Wraps bound around current objects
   // Always wrap = tighter fits
   printf("Depth:%d------", depth);
   for (int i = 0; i < depth; i++)
      printf("------");
   printf("Objects = %d\n\n", (int)objects.size());

   if (objects.size() <= 20)
      return;

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

   int newAxis = axis == 'x' ? 'y' : (axis == 'y' ? 'z' : 'x');

   // Murder the children who don't have healthy siblings.
   if (leftObjects.size() != objects.size() &&
       rightObjects.size() != objects.size()) {
      Left = new BSP(depth + 1, newAxis, leftObjects);
      Right = new BSP(depth + 1, newAxis, rightObjects);
   } else {
      // To indicate dead end and use parents' objects
      Left = Right = NULL;
   }
}

// Given a ray a bounding box, determine whether the ray intersects
// Fuck all this logic, visit Graveyard.cpp to see some of my agony
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
   double newDist = sqrt(pow(distV.x, 2) +
                         pow(distV.y, 2) +
                         pow(distV.z, 2));
   *dist = newDist;
   return true;
}

Intersection BSP::getClosestIntersection(const Ray& ray) {
   double distL, distR;
   bool intersectL = false, intersectR = false;

   if (Left && Right) {
      // There are children! See if they block the ray
      intersectL = intersectAABB(ray, Left->bounds, &distL);
      intersectR = intersectAABB(ray, Right->bounds, &distR);


      // If both hit, follow nearest match
      if (intersectL && intersectR) {
         if (distL < distR) {
            return (*Left).getClosestIntersection(ray);
         }
         return (*Right).getClosestIntersection(ray);
      }

      if (intersectL) {
         return (*Left).getClosestIntersection(ray);
      }

      if (intersectR) {
         return (*Right).getClosestIntersection(ray);
      }
   }

   // No Children. Objects were not divided
   // if (depth > 1)
   //    printf("Done At level: %d, Size: %d\n", depth, (int)objects.size());
   Intersection closestIntersection(false);
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
