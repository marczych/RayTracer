#ifndef __TYPES_H__
#define __TYPES_H__

/* Color struct */
typedef struct color_struct {
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"
} color_t;

typedef struct point_struct {
   int x;
   int y;
} point_t;

#endif
