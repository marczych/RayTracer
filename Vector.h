#ifndef __VECTOR_H__
#define __VECTOR_H__

class Vector {
public:

   double X, Y, Z;

   Vector() : X(0), Y(0), Z(0) {}

   Vector(double in) : X(in), Y(in), Z(in) {}

   Vector(double in_x, double in_y, double in_z) : X(in_x), Y(in_y), Z(in_z) {}

   Vector crossProduct(Vector const & v) const;

   float dotProduct(Vector const & v) const;

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
