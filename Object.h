#ifndef __OBJECT_H__
#define __OBJECT_H__

class Intersection;
class Ray;

/**
 * Base class for all objects that can be ray traced.
 */
class Object {
public:
   virtual Intersection intersect(Ray) = 0;
   virtual double getShininess() = 0;
};

#endif
