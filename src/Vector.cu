#include <math.h>
#include "Vector.h"

__device__ Vector Vector::normalize() {
   return (*this) /= this->length();
}

__device__ Vector Vector::cross(Vector const & v) const {
   return Vector(y*v.z - v.y*z, v.x*z - x*v.z, x*v.y - v.x*y);
}

__device__ double Vector::dot(Vector const & v) const {
   return x*v.x + y*v.y + z*v.z;
}

__device__ double Vector::length() const {
   return sqrtf(x*x + y*y + z*z);
}

__device__ Vector Vector::operator + (Vector const & v) const {
   return Vector(x+v.x, y+v.y, z+v.z);
}

__device__ Vector & Vector::operator += (Vector const & v) {
   x += v.x;
   y += v.y;
   z += v.z;

   return * this;
}

__device__ Vector Vector::operator - (Vector const & v) const {
   return Vector(x-v.x, y-v.y, z-v.z);
}

__device__ Vector & Vector::operator -= (Vector const & v) {
   x -= v.x;
   y -= v.y;
   z -= v.z;

   return * this;
}

__device__ Vector Vector::operator * (Vector const & v) const {
   return Vector(x*v.x, y*v.y, z*v.z);
}

__device__ Vector & Vector::operator *= (Vector const & v) {
   x *= v.x;
   y *= v.y;
   z *= v.z;

   return * this;
}

__device__ Vector Vector::operator / (Vector const & v) const {
   return Vector(x/v.x, y/v.y, z/v.z);
}

__device__ Vector & Vector::operator /= (Vector const & v) {
   x /= v.x;
   y /= v.y;
   z /= v.z;

   return * this;
}

__device__ Vector Vector::operator * (double const s) const {
   return Vector(x*s, y*s, z*s);
}

__device__ Vector & Vector::operator *= (double const s) {
   x *= s;
   y *= s;
   z *= s;

   return * this;
}

__device__ Vector Vector::operator / (double const s) const {
   return Vector(x/s, y/s, z/s);
}

__device__ Vector & Vector::operator /= (double const s) {
   x /= s;
   y /= s;
   z /= s;

   return * this;
}
