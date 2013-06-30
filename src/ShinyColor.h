#ifndef __SHINY_COLOR_H__
#define __SHINY_COLOR_H__

#include "Material.h"

class ShinyColor : public Material {
public:
   Color color;
   double shininess;
   double reflectivity;

   ShinyColor(std::istream&);

   virtual Color getColor(Vector);
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
