#ifndef __FLAT_COLOR_H__
#define __FLAT_COLOR_H__

#include "Material.h"
#include "Color.h"

class FlatColor : public Material {
public:
   Color color;
   double shininess;
   double reflectivity;

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
