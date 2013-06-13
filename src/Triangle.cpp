#include "Triangle.h"
#include "Material.h"
#include <math.h>

Intersection Triangle::intersect(Ray ray) {
  Vector e1, e2, h, s, q, normal;
  float a, f, u, v, distance;

  e1 = Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
  e2 = Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

  normal = ((v1 - v0).cross(v2 - v0)).normalize();

  h = ray.direction.cross(e2);
  a = e1.dot(h);

  if (a > -0.00001 && a < 0.00001)
    return Intersection(false);

  f = 1 / a;
  s = Vector(ray.origin.x - v0.x,
   ray.origin.y - v0.y, ray.origin.z - v0.z);

  u = f * s.dot(h);

  if (u < 0.0 || u > 1.0)
    return Intersection(false);

  q = s.cross(e1);
  v = f * ray.direction.dot(q);

  if (v < 0.0 || u + v > 1.0)
    return Intersection(false);

  distance = f * e2.dot(q);

  // Ray Intersection
  if (distance > 0.00001) {
    Vector point = ray.origin + Vector(distance) * ray.direction;
    return Intersection(ray, point, distance, normal, ray.material, material, this);
  }

  //Line Intersection, Not Ray
  else
     return Intersection(false);
}

Boundaries Triangle::getBounds() {
   return bounds;
}
