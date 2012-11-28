#ifndef __COLOR_H__
#define __COLOR_H__

class Color {
public:
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"

   Color() : r(0.0), g(0.0), b(0.0) {}
   Color(double r_, double g_, double b_) : r(r_), g(g_), b(b_), f(1.0) {}
   Color(double r_, double g_, double b_, double f_) : r(r_), g(g_), b(b_), f(f_) {}
};

#endif
