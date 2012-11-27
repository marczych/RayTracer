#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Image.h"

/**
 * RayTracer main.
 */
int main(void) {
   Image image(500, 500);

   image.pixel(250, 250, Color(0.5, 0, 0, 1));

   image.WriteTga((char *)"awesome.tga", true);

   return 0;
}
