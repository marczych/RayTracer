#include "Sphere.h"
#include <stdio.h>

Intersection Sphere::intersect(Ray ray) {
   /**
    * TODO: Actually perform intersection.
    */
   if (ray.origin.X > 250) {
      return Intersection(true);
   } else {
      return Intersection(Vector(0, 0, 0));
   }
}
