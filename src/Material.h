#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#define NOT_SHINY -1
#define NOT_REFLECTIVE -1

class Vector;
class Color;

class Material {
public:
   virtual Color getColor(Vector) = 0;
   virtual double getShininess();
   virtual double getReflectivity();
};

#endif
