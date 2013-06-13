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

using namespace std;

class BSP {

public:
   int depth;
   char axis;
   Boundaries bounds;
   vector<Object*> objects;
   BSP* Left;
   BSP* Right;

   BSP(vector<Object*> objects_) : objects(objects_) { }

   BSP(int depth_, char axis_, vector<Object*> objects_) :
      depth(depth_), axis(axis_), objects(objects_) {
      build();
   }


   virtual ~BSP() {
      if (Left) {
         delete Left;
      }
      if (Right) {
         delete Right;
      }
   }

   void build();
   bool intersectAABB(const Ray&, Boundaries, double*);
   Intersection getClosestIntersection(const Ray&);
};

#endif
