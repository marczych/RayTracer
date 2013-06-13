#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <math.h>
#include <algorithm>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"
#include "PerlinNoise.h"
#include "Boundaries.h"

class Material;

class Triangle : public Object {
private:
   PerlinNoise perlin;

public:
   Vector v0, v1, v2;
   Material* material;
   Boundaries bounds;

   Triangle(Vector v0_, Vector v1_, Vector v2_, Material* material_) :
    v0(v0_), v1(v1_), v2(v2_), material(material_) {
      bounds.min = Vector(min(v0.x, min(v1.x, v2.x)),
                          min(v0.y, min(v1.y, v2.y)),
                          min(v0.z, min(v1.z, v2.z)));

      bounds.max = Vector(max(v0.x, max(v1.x, v2.x)),
                          max(v0.y, max(v1.y, v2.y)),
                          max(v0.z, max(v1.z, v2.z)));
   }

   virtual Intersection intersect(Ray);
   virtual Boundaries getBounds();

private:
   Color getColor(Vector);
};

#endif
