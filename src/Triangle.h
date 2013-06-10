#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <math.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"
#include "PerlinNoise.h"

class Material;

class Triangle : public Object {
private:
   PerlinNoise perlin;

public:
   Vector v0, v1, v2;
   Material* material;

   Triangle(Vector v0_, Vector v1_, Vector v2_, Material* material_) :
     v0(v0_), v1(v1_), v2(v2_), material(material_) { }

   virtual Intersection intersect(Ray);

private:
   Color getColor(Vector);
};

#endif
