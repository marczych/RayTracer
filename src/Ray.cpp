#include "Ray.h"

/**
 * Calculates the fractional direction for the ray to avoid doing it multiple times.
 */
void Ray::calcFracDirection() {
   fracDir.x = 1.0f / direction.x;
   fracDir.y = 1.0f / direction.y;
   fracDir.z = 1.0f / direction.z;
}
