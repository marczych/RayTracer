#ifndef __BSP_H__
#define __BSP_H__

#include <math.h>
#include <vector>
#include "Boundaries.h"
#include "Ray.h"

class Object;
class Intersection;

#define MIN_OBJECT_COUNT 20

class BSP {
private:
   int depth;
   int axisRetries;
   char axis;
   Boundaries bounds;
   std::vector<Object*> objects;
   BSP* left;
   BSP* right;

   void build();
   char toggleAxis();

public:

   BSP(int depth_, char axis_, std::vector<Object*> objects_) :
    depth(depth_), axis(axis_), objects(objects_) {
      axisRetries = 0;
      left = right = NULL;
      build();
   }

   Intersection getClosestIntersection(const Ray&);
   Intersection getClosestObjectIntersection(const Ray&);
};

#endif
