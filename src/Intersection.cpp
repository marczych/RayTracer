#include "Intersection.h"

Color Intersection::getColor() const {
   return material->getColor(intersection);
}
