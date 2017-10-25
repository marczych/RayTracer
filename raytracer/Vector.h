#ifndef __VECTOR_H__
#define __VECTOR_H__

class Vector {
public:

   double x, y, z;

   Vector() : x(0), y(0), z(0) {}

   Vector(double in) : x(in), y(in), z(in) {}

   Vector(double in_x, double in_y, double in_z) : x(in_x), y(in_y), z(in_z) {}

   Vector normalize();

   Vector cross(Vector const & v) const;

   double dot(Vector const & v) const;

   double length() const;

   Vector operator + (Vector const & v) const;

   Vector & operator += (Vector const & v);

   Vector operator - (Vector const & v) const;

   Vector & operator -= (Vector const & v);

   Vector operator * (Vector const & v) const;

   Vector & operator *= (Vector const & v);

   Vector operator / (Vector const & v) const;

   Vector & operator /= (Vector const & v);

   Vector operator * (double const s) const;

   Vector & operator *= (double const s);

   Vector operator / (double const s) const;

   Vector & operator /= (double const s);
};

#endif
