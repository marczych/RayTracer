#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"
#include "PerlinNoise.h"
#include "Boundaries.h"

class Material;

class Sphere : public Object {
private:
   PerlinNoise perlin;

public:
   Vector center;
   double radius;
   Material* material;
   Boundaries bounds;

   Sphere(Vector center_, double radius_, Material* material_) : center(center_),
    radius(radius_), material(material_)  {
     bounds.min = center - Vector(radius, radius, radius);
     bounds.max = center + Vector(radius, radius, radius);
   }

   virtual Intersection intersect(Ray);
   virtual Boundaries getBounds();

private:
   Color getColor(Vector);
};

#endif
