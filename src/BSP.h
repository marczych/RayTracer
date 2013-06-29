#ifndef __BSP_H__
#define __BSP_H__

#include <math.h>
#include <limits>
#include "Vector.h"
#include <vector>
#include "Ray.h"
#include "Object.h"
#include "Intersection.h"
#include "Boundaries.h"

#define MIN_OBJECT_COUNT 20

class BSP {
private:
   int depth;
   char axis;
   Boundaries bounds;
   vector<Object*> objects;
   BSP* left;
   BSP* right;

   bool intersectAABB(const Ray&, Boundaries, double*);
   void build();

public:

   BSP(vector<Object*> objects_) : objects(objects_) { }

   BSP(int depth_, char axis_, vector<Object*> objects_) :
      depth(depth_), axis(axis_), objects(objects_) {
      build();
   }

   virtual ~BSP() {
      if (left) {
         delete left;
      }
      if (right) {
         delete right;
      }
   }

   Intersection getClosestIntersection(const Ray&);
};

#endif
