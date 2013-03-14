#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "Vector.h"

class Camera {
public:
   Vector position;
   Vector up;
   Vector lookAt;
   Vector w, u, v;
   double screenWidth;

   Camera() {
      position = Vector(0.0, 0.0, 100.0);
      up = Vector(0.0, 1.0, 0.0);
      lookAt = Vector(0.0, 0.0, 0.0);
      screenWidth = 1000;

      calculateWUV();
   }

   Camera(Vector position_, Vector up_, Vector lookAt_, double screenWidth_) :
      position(position_), up(up_), lookAt(lookAt_), screenWidth(screenWidth_) {
       calculateWUV();
   }

   void calculateWUV();
};

#endif
