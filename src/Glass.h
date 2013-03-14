#ifndef __GLASS_H__
#define __GLASS_H__

#include <iostream>
#include "Material.h"
#include "Color.h"

class Glass : public Material {
private:
   double refractiveIndex;
   double shininess;

public:
   Glass(std::istream&);

   virtual Color getColor(Vector);
   virtual double getRefractiveIndex();
   virtual double getShininess();
};

#endif
