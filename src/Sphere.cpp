#include "Sphere.h"
#include "Material.h"
#include <math.h>

Intersection Sphere::intersect(Ray ray) {
   Vector deltap = ray.origin - center;
   double a = ray.direction.dot(ray.direction);
   double b = deltap.dot(ray.direction) * 2;
   double c = deltap.dot(deltap) - (radius * radius);

   double disc = b * b - 4 * a * c;
   if (disc < 0) {
      return Intersection(false); // No intersection.
   }

   disc = sqrt(disc);

   double q;
   if (b < 0) {
      q = (-b - disc) * 0.5;
   } else {
      q = (-b + disc) * 0.5;
   }

   double r1 = q / a;
   double r2 = c / q;

   if (r1 > r2) {
      double tmp = r1;
      r1 = r2;
      r2 = tmp;
   }

   double distance = r1;
   if (distance < 0) {
      distance = r2;
   }

   if (distance < 0 || isnan(distance)) {
      return Intersection(false); // No intersection.
   }

   Vector point = ray.origin + (ray.direction * distance);
   Vector normal = (point - center).normalize();

   normal = material->modifyNormal(normal, point);

   // Normal needs to be flipped if this is a refractive ray.
   if (ray.direction.dot(normal) > 0) {
      normal = normal * -1;
   }

   return Intersection(ray, point, distance, normal, ray.material, material, this);
}

Boundaries Sphere::getBounds() {
   return bounds;
}
