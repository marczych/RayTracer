#include "Camera.h"

void Camera::calculateWUV() {
   w = (lookAt - position).normalize();
   u = up.cross(w).normalize();
   v = w.cross(u);
}
