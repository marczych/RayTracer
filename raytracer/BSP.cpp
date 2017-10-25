#include <limits>

#include "BSP.h"
#include "Intersection.h"
#include "objects/Object.h"

using namespace std;

void BSP::build() {
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

   // For debugging.
   if (false) {
      for (int i = 0; i < depth; i++) {
        cout << "\t";
      }
      cout << objects.size() << " | " <<
         bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << " || " <<
         bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z <<
         " % " << axis << " X " << axisRetries << endl;
   }

   // We've hit our limit so this is a leaf node. No need to split again.
   if (objects.size() <= MIN_OBJECT_COUNT) {
      return;
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
   } else if (axisRetries < 2) {
      axis = toggleAxis();
      axisRetries++;
      build();
   } else {
      // Do nothing since we're out of axis retries.
   }
}

char BSP::toggleAxis() {
   return axis == 'x' ? 'y' : (axis == 'y' ? 'z' : 'x');
}

Intersection BSP::getClosestIntersection(const Ray& ray) {
   double distance;
   if (!bounds.intersect(ray, &distance)) {
      return Intersection();
   }

   if (left != NULL && right != NULL) {
      Intersection leftIntersection = left->getClosestIntersection(ray);
      Intersection rightIntersection = right->getClosestIntersection(ray);

      return leftIntersection.distance < rightIntersection.distance ?
       leftIntersection : rightIntersection;
   } else {
      return getClosestObjectIntersection(ray);
   }
}

Intersection BSP::getClosestObjectIntersection(const Ray& ray) {
   // No children so just go through current objects like normal.
   Intersection closestIntersection;

   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect && intersection.distance <
       closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection;
}
