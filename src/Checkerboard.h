#ifndef __CHECKERBOARD_H__
#define __CHECKERBOARD_H__

#include "Material.h"
#include "Color.h"

class Checkerboard : public Material {
public:
   Color color1;
   Color color2;
   double scale;
   double shininess;
   double reflectivity;

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
