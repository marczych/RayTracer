#include "Intersection.h"

Color Intersection::getColor() {
   return material->getColor(intersection);
}
