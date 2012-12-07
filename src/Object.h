#ifndef __OBJECT_H__
#define __OBJECT_H__

#define NOT_SHINY -1
#define NOT_REFLECTIVE -1

class Intersection;
class Ray;

/**
 * Base class for all objects that can be ray traced.
 */
class Object {
public:
   virtual Intersection intersect(Ray) = 0;
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
