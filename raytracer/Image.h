/**
 * Bob Somers
 * rsomers@calpoly.edu
 * CPE 473, Winter 2010
 * Cal Poly, San Luis Obispo
 */

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Color.h"

class Image {
public:
    Image(int width, int height);
    ~Image();

    // if scale_color is true, the output targa will have its color space scaled
    // to the global max, otherwise it will be clamped at 1.0
    void WriteTga(const char *outfile, bool scale_color = true);

    void GenTestPattern();

    // property accessors
    Color pixel(int x, int y);
    void pixel(int x, int y, Color pxl);
    int width() const { return _width; }
    int height() const { return _height; }
    double max() const { return _max; }

private:
    int _width;
    int _height;
    Color *_pixmap;
    double _max;
};

#endif
