#ifndef __CHECKERBOARD_H__
#define __CHECKERBOARD_H__

#include "Material.h"

class Checkerboard : public Material {
   virtual Color getColor(Vector);
};

#endif
