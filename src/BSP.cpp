#include "BSP.h"

using namespace std;

void BSP::build() {
   // Wraps bound around current objects
   // Always wrap = tighter fits
   printf("D:%d      ", depth);
   for (int i = 0; i < depth; i++)
      printf("      ");
   printf("Size = %d\n\n", (int)objects.size());

   if (objects.size() <= 20)
      return;

      for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
         Boundaries curr = (*itr)->getBounds();
         bounds.min = Vector(min(bounds.min.x, curr.min.x),
                             min(bounds.min.y, curr.min.y),
                             min(bounds.min.z, curr.min.z));
         bounds.max = Vector(max(bounds.max.x, curr.max.x),
                             max(bounds.max.y, curr.max.y),
                             max(bounds.max.z, curr.max.z));
      }

   switch(axis) {
      case 'x': //printf("Axis %c, Min %f, Max %f\n", axis, bounds.min.x, bounds.max.x);
                break;
      case 'y': //printf("Axis %c, Min %f, Max %f\n", axis, bounds.min.y, bounds.max.y);
                break;
      case 'z': //printf("Axis %c, Min %f, Max %f\n", axis, bounds.min.z, bounds.max.z);
                break;
   }
   Boundaries l = Boundaries();
   Boundaries r = Boundaries();
   double splitValue = bounds.splitValue(axis);
   bounds.split(&l, &r, axis);

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

      if (min < splitValue)
         leftObjects.push_back(obj);

      if (max > splitValue)
         rightObjects.push_back(obj);
   }

   int newAxis = axis == 'x' ? 'y' : (axis == 'y' ? 'z' : 'x');

   // Abort the children
   //if (leftObjects.size() != objects.size())
   //   Left = new BSP(depth + 1, newAxis, leftObjects);
   //if (rightObjects.size() != objects.size())
   //   Right = new BSP(depth + 1, newAxis, rightObjects);
   if (leftObjects.size() != objects.size() &&
       rightObjects.size() != objects.size()) {
      Left = new BSP(depth + 1, newAxis, leftObjects);
      Right = new BSP(depth + 1, newAxis, rightObjects);
   }
}

Intersection getClosestIntersection(Ray& ray) {
   
}

