#include "Intersection.h"

Color Intersection::getColor() const {
   return endMaterial->getColor(intersection);
}
