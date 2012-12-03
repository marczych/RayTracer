#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"

class Sphere : public Object {
public:
   Vector center;
   double radius;
   Color color;
   double shininess;

   Sphere(Vector center_, double radius_, Color color_, double shininess_) :
    center(center_), radius(radius_), color(color_), shininess(shininess_) {}

   virtual Intersection intersect(Ray);
   virtual double getShininess();
};

#endif
