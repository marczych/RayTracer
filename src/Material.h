#ifndef __MATERIAL_H__
#define __MATERIAL_H__

class Vector;
class Color;

class Material {
public:
   virtual Color getColor(Vector) = 0;
};

#endif
