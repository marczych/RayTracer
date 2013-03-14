#ifndef __FLAT_COLOR_H__
#define __FLAT_COLOR_H__

#include <iostream>
#include "Material.h"
#include "Color.h"

class FlatColor : public Material {
private:
   Color color;

public:
   FlatColor(std::istream&);

   virtual Color getColor(Vector);
};

#endif
