#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Object.h"
#include "raytracer/Boundaries.h"
#include "raytracer/Intersection.h"
#include "raytracer/Ray.h"
#include "raytracer/Vector.h"

class Material;

class Sphere : public Object {
public:
   Vector center;
   double radius;
   Material* material;
   Boundaries bounds;

   Sphere(Vector center_, double radius_, Material* material_) : center(center_),
    radius(radius_), material(material_) {
      bounds.min = center - Vector(radius, radius, radius);
      bounds.max = center + Vector(radius, radius, radius);
   }

   virtual Intersection intersect(Ray);
   virtual Boundaries getBounds();

private:
   Color getColor(Vector);
};

#endif
