#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <cuda_runtime_api.h>
#include <cuda.h>

class Vector {
public:

   double x, y, z;

   __device__ Vector() : x(0), y(0), z(0) {}

   __device__ Vector(double in) : x(in), y(in), z(in) {}

   __device__ Vector(double in_x, double in_y, double in_z) : x(in_x), y(in_y), z(in_z) {}

   __device__ Vector normalize();

   __device__ Vector cross(Vector const & v) const;

   __device__ double dot(Vector const & v) const;

   __device__ double length() const;

   __device__ Vector operator + (Vector const & v) const;

   __device__ Vector & operator += (Vector const & v);

   __device__ Vector operator - (Vector const & v) const;

   __device__ Vector & operator -= (Vector const & v);

   __device__ Vector operator * (Vector const & v) const;

   __device__ Vector & operator *= (Vector const & v);

   __device__ Vector operator / (Vector const & v) const;

   __device__ Vector & operator /= (Vector const & v);

   __device__ Vector operator * (double const s) const;

   __device__ Vector & operator *= (double const s);

   __device__ Vector operator / (double const s) const;

   __device__ Vector & operator /= (double const s);
};

#endif
