#ifndef __FLAT_COLOR_H__
#define __FLAT_COLOR_H__

#include "Material.h"

class FlatColor : public Material {
   virtual Color getColor(Vector);
};

#endif
