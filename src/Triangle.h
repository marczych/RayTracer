#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <math.h>
#include <stdlib.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"
#include "Boundaries.h"

class Material;

class Triangle : public Object {
public:
   Vector v0, v1, v2;
   Material* material;
   Boundaries bounds;

   Triangle(Vector v0_, Vector v1_, Vector v2_, Material* material_) :
    v0(v0_), v1(v1_), v2(v2_), material(material_) {
      bounds.min = Vector(std::min(v0.x, std::min(v1.x, v2.x)),
                          std::min(v0.y, std::min(v1.y, v2.y)),
                          std::min(v0.z, std::min(v1.z, v2.z)));

      bounds.max = Vector(std::max(v0.x, std::max(v1.x, v2.x)),
                          std::max(v0.y, std::max(v1.y, v2.y)),
                          std::max(v0.z, std::max(v1.z, v2.z)));
   }

   virtual Intersection intersect(Ray);
   virtual Boundaries getBounds();

private:
   Color getColor(Vector);
};

#endif
