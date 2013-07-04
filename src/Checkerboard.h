#ifndef __CHECKERBOARD_H__
#define __CHECKERBOARD_H__

#include "Material.h"

class Checkerboard : public Material {
private:
   Color color1;
   Color color2;
   double scale;
   double shininess;
   double reflectivity;

public:
   Checkerboard(std::istream&);

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
