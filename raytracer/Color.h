#ifndef __COLOR_H__
#define __COLOR_H__

/**
 * "NaN To Zero"
 * Converts a NaN value to zero. Otherwise adding values to NaN results in NaN.
 * When adding colors together we usually want to ignore it and use 0 instead.
 */
#define NTZ(X) (isnan((X)) ? 0.0 : (X))

#include <math.h>

class Color {
public:
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"

   Color() : r(0.0), g(0.0), b(0.0), f(1.0) {}
   Color(double r_, double g_, double b_) : r(r_), g(g_), b(b_), f(1.0) {}
   Color(double r_, double g_, double b_, double f_) : r(r_), g(g_), b(b_), f(f_) {}

   Color operator + (Color const & c) const;
   Color operator * (double amount) const;
};

#endif
