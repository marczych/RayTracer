#ifndef __PERLIN_NOISE_H__
#define __PERLIN_NOISE_H__

/**
 * Taken from: http://www.codermind.com/articles/Raytracer-in-C++-Part-III-Textures.html
 */

class PerlinNoise {
private:
   int p[512];

   double fade(double);
   double lerp(double, double, double);
   double grad(int, double, double, double);

public:
   PerlinNoise();

   double noise(double, double, double);
};

#endif
