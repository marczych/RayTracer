#include <math.h>
#include "Vector.h"

Vector Vector::crossProduct(Vector const & v) const {
   return Vector(Y*v.Z - v.Y*Z, v.X*Z - X*v.Z, X*v.Y - v.X*Y);
}

float Vector::dotProduct(Vector const & v) const {
   return X*v.X + Y*v.Y + Z*v.Z;
}

double Vector::length() const {
   return sqrtf(X*X + Y*Y + Z*Z);
}

Vector Vector::operator + (Vector const & v) const {
   return Vector(X+v.X, Y+v.Y, Z+v.Z);
}

Vector & Vector::operator += (Vector const & v) {
   X += v.X;
   Y += v.Y;
   Z += v.Z;

   return * this;
}

Vector Vector::operator - (Vector const & v) const {
   return Vector(X-v.X, Y-v.Y, Z-v.Z);
}

Vector & Vector::operator -= (Vector const & v) {
   X -= v.X;
   Y -= v.Y;
   Z -= v.Z;

   return * this;
}

Vector Vector::operator * (Vector const & v) const {
   return Vector(X*v.X, Y*v.Y, Z*v.Z);
}

Vector & Vector::operator *= (Vector const & v) {
   X *= v.X;
   Y *= v.Y;
   Z *= v.Z;

   return * this;
}

Vector Vector::operator / (Vector const & v) const {
   return Vector(X/v.X, Y/v.Y, Z/v.Z);
}

Vector & Vector::operator /= (Vector const & v) {
   X /= v.X;
   Y /= v.Y;
   Z /= v.Z;

   return * this;
}

Vector Vector::operator * (double const s) const {
   return Vector(X*s, Y*s, Z*s);
}

Vector & Vector::operator *= (double const s) {
   X *= s;
   Y *= s;
   Z *= s;

   return * this;
}

Vector Vector::operator / (double const s) const {
   return Vector(X/s, Y/s, Z/s);
}

Vector & Vector::operator /= (double const s) {
   X /= s;
   Y /= s;
   Z /= s;

   return * this;
}
