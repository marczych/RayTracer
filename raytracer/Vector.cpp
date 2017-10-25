#include <math.h>
#include "Vector.h"

Vector Vector::normalize() {
   return (*this) /= this->length();
}

Vector Vector::cross(Vector const & v) const {
   return Vector(y*v.z - v.y*z, v.x*z - x*v.z, x*v.y - v.x*y);
}

double Vector::dot(Vector const & v) const {
   return x*v.x + y*v.y + z*v.z;
}

double Vector::length() const {
   return sqrtf(x*x + y*y + z*z);
}

Vector Vector::operator + (Vector const & v) const {
   return Vector(x+v.x, y+v.y, z+v.z);
}

Vector & Vector::operator += (Vector const & v) {
   x += v.x;
   y += v.y;
   z += v.z;

   return * this;
}

Vector Vector::operator - (Vector const & v) const {
   return Vector(x-v.x, y-v.y, z-v.z);
}

Vector & Vector::operator -= (Vector const & v) {
   x -= v.x;
   y -= v.y;
   z -= v.z;

   return * this;
}

Vector Vector::operator * (Vector const & v) const {
   return Vector(x*v.x, y*v.y, z*v.z);
}

Vector & Vector::operator *= (Vector const & v) {
   x *= v.x;
   y *= v.y;
   z *= v.z;

   return * this;
}

Vector Vector::operator / (Vector const & v) const {
   return Vector(x/v.x, y/v.y, z/v.z);
}

Vector & Vector::operator /= (Vector const & v) {
   x /= v.x;
   y /= v.y;
   z /= v.z;

   return * this;
}

Vector Vector::operator * (double const s) const {
   return Vector(x*s, y*s, z*s);
}

Vector & Vector::operator *= (double const s) {
   x *= s;
   y *= s;
   z *= s;

   return * this;
}

Vector Vector::operator / (double const s) const {
   return Vector(x/s, y/s, z/s);
}

Vector & Vector::operator /= (double const s) {
   x /= s;
   y /= s;
   z /= s;

   return * this;
}
