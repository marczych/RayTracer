#include "Material.h"
#include "NormalMap.h"

/**
 * Put Material code here!
 */

double Material::getShininess() {
   return NOT_SHINY;
}

double Material::getReflectivity() {
   return NOT_REFLECTIVE;
}

double Material::getRefractiveIndex() {
   return NOT_REFRACTIVE;
}

Vector Material::modifyNormal(const Vector& normal, const Vector& point) {
   if (normalMap != NULL) {
      return normalMap->modifyNormal(normal, point);
   } else {
      return normal;
   }
}
