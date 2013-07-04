#ifndef __AIR_H__
#define __AIR_H__

#include "Material.h"

class Air : public Material {
public:
   Air() {}

   virtual Color getColor(Vector);
   virtual double getRefractiveIndex();
};

#endif
