/**
 * Bob Somers
 * rsomers@calpoly.edu
 * CPE 473, Winter 2010
 * Cal Poly, San Luis Obispo
 */

#include "Image.h"

Image::Image(int width, int height)
{
    _width = width;
    _height = height;
    _max = 1.0;

    _pixmap = (Color*)malloc(sizeof(Color) * _width * _height);
}

Image::~Image()
{
    free(_pixmap);
}

void Image::WriteTga(const char *outfile, bool scale_color)
{
    FILE *fp = fopen(outfile, "w");
    if (fp == NULL)
    {
        perror("ERROR: Image::WriteTga() failed to open file for writing!\n");
        exit(EXIT_FAILURE);
    }
    
    // write 24-bit uncompressed targa header
    // thanks to Paul Bourke (http://local.wasp.uwa.edu.au/~pbourke/dataformats/tga/)
    putc(0, fp);
    putc(0, fp);
    
    putc(2, fp); // type is uncompressed RGB
    
    putc(0, fp);
    putc(0, fp);
    putc(0, fp);
    putc(0, fp);
    putc(0, fp);
    
    putc(0, fp); // x origin, low byte
    putc(0, fp); // x origin, high byte
    
    putc(0, fp); // y origin, low byte
    putc(0, fp); // y origin, high byte

    putc(_width & 0xff, fp); // width, low byte
    putc((_width & 0xff00) >> 8, fp); // width, high byte

    putc(_height & 0xff, fp); // height, low byte
    putc((_height & 0xff00) >> 8, fp); // height, high byte

    putc(24, fp); // 24-bit color depth

    putc(0, fp);

    // write the raw pixel data in groups of 3 bytes (BGR order)
    for (int y = 0; y < _height; y++)
    {
        for (int x = 0; x < _width; x++)
        {
            // if color scaling is on, scale 0.0 -> _max as a 0 -> 255 unsigned byte
            unsigned char rbyte, gbyte, bbyte;
            Color* color = _pixmap + (x * _height + y);
            if (scale_color)
            {
                rbyte = (unsigned char)((color->r / _max) * 255);
                gbyte = (unsigned char)((color->g / _max) * 255);
                bbyte = (unsigned char)((color->b / _max) * 255);
            }
            else
            {
                double r = (color->r > 1.0) ? 1.0 : color->r;
                double g = (color->g > 1.0) ? 1.0 : color->g;
                double b = (color->b > 1.0) ? 1.0 : color->b;
                rbyte = (unsigned char)(r * 255);
                gbyte = (unsigned char)(g * 255);
                bbyte = (unsigned char)(b * 255);
            }
            putc(bbyte, fp);
            putc(gbyte, fp);
            putc(rbyte, fp);
        }
    }

    fclose(fp);
}

void Image::GenTestPattern()
{
    Color pxl(0.0, 0.0, 0.0, 0.0);
    int i, j, color;
    float radius, dist;
    
    // draw a rotating color checkerboard (RGB) in a 25x25 pixel grid
    for (int x = 0; x < _width; x++)
    {
        for (int y = 0; y < _height; y++)
        {
            i = x / 25;
            j = y / 25;
            color = (i + j) % 3;
            
            switch (color)
            {
                case 0: // red
                    pxl.r = 1.0; pxl.g = 0.0; pxl.b = 0.0;
                    break;

                case 1: // green
                    pxl.r = 0.0; pxl.g = 1.0; pxl.b = 0.0;
                    break;

                case 2: // blue
                    pxl.r = 0.0; pxl.g = 0.0; pxl.b = 1.0;
                    break;
            }

            pixel(x, y, pxl);
        } 
    }

    // draw a black circle in the top left quadrant (centered at (i, j))
    pxl.r = 0.0; pxl.g = 0.0; pxl.b = 0.0;
    i = _width / 4;
    j = 3 * _height / 4;
    radius = (((float)_width / 4.0) < ((float)_height / 4.0)) ? (float)_width / 4.0 : (float)_height / 4.0;
    for (int x = 0; x < _width; x++)
    {
        for (int y = 0; y < _height; y++)
        {
            dist = sqrtf((float)((x - i) * (x - i)) + (float)((y - j) * (y - j)));
            if (dist <= (float)radius)
            {
                pixel(x, y, pxl);
            }
        }
    }
    
    // draw a white circle in the lower right quadrant (centered at (i, j))
    pxl.r = 1.0; pxl.g = 1.0; pxl.b = 1.0;
    i = 3 * _width / 4;
    j = _height / 4;
    radius = (((float)_width / 4.0) < ((float)_height / 4.0)) ? (float)_width / 4.0 : (float)_height / 4.0;
    for (int x = 0; x < _width; x++)
    {
        for (int y = 0; y < _height; y++)
        {
            dist = sqrtf((float)((x - i) * (x - i)) + (float)((y - j) * (y - j)));
            if (dist <= (float)radius)
            {
                pixel(x, y, pxl);
            }
        }
    }
}

Color Image::pixel(int x, int y)
{
    if (x < 0 || x > _width - 1 ||
        y < 0 || y > _height - 1)
    {
        // catostrophically fail
        fprintf(stderr, "ERROR: Image::pixel(%d, %d) outside range of the image!\n", x, y);
        exit(EXIT_FAILURE);
    }
    
    return _pixmap[x * _height + y];
}

void Image::pixel(int x, int y, Color pxl)
{
    if (x < 0 || x > _width - 1 ||
        y < 0 || y > _height - 1)
    {
        // catostrophically fail
        fprintf(stderr, "ERROR: Image::pixel(%d, %d, pixel) outside range of the image!\n", x, y);
        exit(EXIT_FAILURE);
    }
    
    _pixmap[x * _height + y] = pxl;

    // update the max color if necessary
    _max = (pxl.r > _max) ? pxl.r : _max;
    _max = (pxl.g > _max) ? pxl.g : _max;
    _max = (pxl.b > _max) ? pxl.b : _max;
}

