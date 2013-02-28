#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include "Vector.h"
#include "Ray.h"
#include "Intersection.h"
#include "Object.h"
#include "Material.h"

class Sphere : public Object {
public:
   Vector center;
   double radius;
   Material* material;
   double shininess;
   double reflectivity;

   Sphere(Vector center_, double radius_, Material* material_, double shininess_,
    double reflectivity_) : center(center_), radius(radius_), material(material_),
    shininess(shininess_), reflectivity(reflectivity_) {}

   virtual Intersection intersect(Ray);
   virtual double getShininess();
   virtual double getReflectivity();

private:
   Color getColor(Vector);
};

#endif
