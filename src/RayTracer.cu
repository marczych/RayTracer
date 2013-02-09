#include <limits>
#include "RayTracer.h"
#include "Image.h"
#include "Object.h"
#include "Sphere.h"
#include "Intersection.h"
#include "Light.h"

using namespace std;

#define ERROR_HANDLER(x) ErrorHandler(x, __FILE__, __LINE__)
#define TILE_WIDTH 16

static void ErrorHandler(cudaError_t err, const char *file, int line) {
   if (err != cudaSuccess) {
      fprintf(stderr, "%s in line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}

RayTracer::RayTracer(int width_, int height_, int maxReflections_, int superSamples_,
 int depthComplexity_) : width(width_), height(height_),
 maxReflections(maxReflections_), superSamples(superSamples_),
 imageScale(1), depthComplexity(depthComplexity_), dispersion(5.0f), raysCast(0) {}

RayTracer::~RayTracer() {
}

void RayTracer::traceRays(uchar4* devImage, Sphere* devSpheres, Light* devLights) {
   RayTracer* devRayTracer;
   // Calculate these on the CPU so they can be accessed on the GPU.
   numSpheres = spheres.size();
   numLights = lights.size();

   // Reset depthComplexity to avoid unnecessary loops.
   if (dispersion < 0) {
      depthComplexity = 1;
   }

   ERROR_HANDLER(cudaMalloc((void**)&devRayTracer, sizeof(RayTracer)));
   ERROR_HANDLER(cudaMemcpy(devRayTracer, this, sizeof(RayTracer), cudaMemcpyHostToDevice));

   int gridWidth = ceil((float)width/TILE_WIDTH);
   int gridHeight = ceil((float)height/TILE_WIDTH);

   dim3 dimGrid(gridWidth, gridHeight);
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

   cudaTraceRays<<<dimGrid, dimBlock>>>(devSpheres, devLights, devImage,
    devRayTracer);

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   ERROR_HANDLER(cudaFree(devRayTracer));
}

__global__ void cudaTraceRays(Sphere* spheres, Light* lights, uchar4* image,
 RayTracer* rayTracer) {
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
   int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

   if (x < rayTracer->width && y < rayTracer->height) {
      Color color = rayTracer->castRayForPixel(x, y, spheres, lights);
      uchar4* imageColor = image + (x * rayTracer->height + y);

      imageColor->x = color.r;
      imageColor->y = color.g;
      imageColor->z = color.b;
      imageColor->w = 255;
   }
}

__device__ Color RayTracer::castRayForPixel(int x, int y, Sphere* spheres,
 Light* lights) {
   double rayX = (x - width / 2)/2.0;
   double rayY = (y - height / 2)/2.0;
   double pixelWidth = rayX - (x + 1 - width / 2)/2.0;
   double sampleWidth = pixelWidth / superSamples;
   double sampleStartX = rayX - pixelWidth/2.0;
   double sampleStartY = rayY - pixelWidth/2.0;
   double sampleWeight = 1.0 / (superSamples * superSamples);
   Color color;

   for (int x = 0; x < superSamples; x++) {
      for (int y = 0; y < superSamples; y++) {
         Vector imagePlanePoint = camera.lookAt -
          (camera.u * (sampleStartX + (x * sampleWidth)) * imageScale) +
          (camera.v * (sampleStartY + (y * sampleWidth)) * imageScale);

         color = color + (castRayAtPoint(imagePlanePoint, spheres, lights) *
          sampleWeight);
      }
   }

   return color;
}

__device__ Color RayTracer::castRayAtPoint(Vector point, Sphere* spheres,
 Light* lights) {
   Color color;

   for (int i = 0; i < depthComplexity; i++) {
      Ray viewRay(camera.position, point - camera.position, maxReflections);

      //if (depthComplexity > 1) {
      //   Vector disturbance(
      //    (dispersion / RAND_MAX) * (1.0f * rand()),
      //    (dispersion / RAND_MAX) * (1.0f * rand()),
      //    0.0f);

      //   viewRay.origin = viewRay.origin + disturbance;
      //   viewRay.direction = point - viewRay.origin;
      //   viewRay.direction = viewRay.direction.normalize();
      //}

      color = color + (castRay(viewRay, spheres, lights) *
       (1 / (float)depthComplexity));
   }

   return color;
}

__device__ Color RayTracer::castRay(Ray ray, Sphere* spheres, Light* lights) {
   Intersection intersection = getClosestIntersection(ray, spheres);

   if (intersection.didIntersect) {
      return performLighting(intersection, lights, spheres);
   } else {
      return Color();
   }
}

__device__ Intersection RayTracer::getClosestIntersection(Ray ray,
 Sphere* spheres) {
   Intersection closestIntersection(false);
   closestIntersection.distance = 983487438;

   for (int i = 0; i < numSpheres; i++) {
      Sphere* sphere = spheres + i;
      Intersection intersection = sphere->intersect(ray);

      if (intersection.didIntersect && intersection.distance <
       closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection;
}

/**
 * Basically same code as getClosestIntersection but short circuits if an
 * intersection closer to the given light distance is found.
 */
__device__ bool RayTracer::isInShadow(Ray ray, Sphere* spheres,
 double lightDistance) {
   for (int i = 0; i < numSpheres; i++) {
      Sphere* sphere = spheres + i;
      Intersection intersection = sphere->intersect(ray);

      if (intersection.didIntersect && intersection.distance <
       lightDistance) {
         return true;
      }
   }

   return false;
}

__device__ Color RayTracer::performLighting(Intersection intersection,
 Light* lights, Sphere* spheres) {
   Color ambientColor = getAmbientLighting(intersection);
   Color diffuseAndSpecularColor = getDiffuseAndSpecularLighting(
    intersection, lights, spheres);
   //Color reflectedColor = getReflectiveLighting(intersection);

   return ambientColor + diffuseAndSpecularColor;// + reflectedColor;
}

__device__ Color RayTracer::getAmbientLighting(Intersection intersection) {
   return intersection.color * 0.2;
}

__device__ Color RayTracer::getDiffuseAndSpecularLighting(
 Intersection intersection, Light* lights, Sphere* spheres) {
   Color diffuseColor(0.0, 0.0, 0.0);
   Color specularColor(0.0, 0.0, 0.0);

   for (int i = 0; i < numLights; i++) {
      Light* light = lights + i;
      Vector lightOffset = light->position - intersection.intersection;
      double lightDistance = lightOffset.length();
      /**
       * TODO: Be careful about normalizing lightOffset too.
       */
      Vector lightDirection = lightOffset.normalize();
      double dotProduct = intersection.normal.dot(lightDirection);

      /**
       * Intersection is facing light.
       */
      if (dotProduct >= 0.0f) {
         Ray shadowRay = Ray(intersection.intersection, lightDirection, 1);
         if (isInShadow(shadowRay, spheres, lightDistance)) {
            /**
             * Position is in shadow of another object - continue with other lights.
             */
            continue;
         }

         diffuseColor = (diffuseColor + (intersection.color * dotProduct)) *
          light->intensity;
         specularColor = specularColor + getSpecularLighting(intersection, light);
      }
   }

   return diffuseColor + specularColor;
}

__device__ Color RayTracer::getSpecularLighting(Intersection intersection, Light* light) {
   Color specularColor(0.0, 0.0, 0.0);
   double shininess = intersection.object->getShininess();

   if (shininess == NOT_SHINY) {
      /* Don't perform specular lighting on non shiny objects. */
      return specularColor;
   }

   Vector view = (intersection.ray.origin - intersection.intersection).normalize();
   Vector lightOffset = light->position - intersection.intersection;
   Vector reflected = reflectVector(lightOffset.normalize(), intersection.normal);

   double dot = view.dot(reflected);

   if (dot <= 0) {
      return specularColor;
   }

   double specularAmount = pow(dot, shininess) * light->intensity;

   specularColor.r = specularAmount;
   specularColor.g = specularAmount;
   specularColor.b = specularAmount;

   return specularColor;
}

//__device__ Color RayTracer::getReflectiveLighting(Intersection intersection) {
//   double reflectivity = intersection.object->getReflectivity();
//   int reflectionsRemaining = intersection.ray.reflectionsRemaining;
//
//   if (reflectivity == NOT_REFLECTIVE || reflectionsRemaining <= 0) {
//      return Color();
//   } else {
//      Vector reflected = reflectVector(intersection.ray.origin, intersection.normal);
//      Ray reflectedRay(intersection.intersection, reflected, reflectionsRemaining - 1);
//
//      return castRay(reflectedRay) * reflectivity;
//   }
//}

__device__ Vector RayTracer::reflectVector(Vector vector, Vector normal) {
   return normal * 2 * vector.dot(normal) - vector;
}

void RayTracer::readScene(istream& in) {
   string type;

   in >> type;

   while (in.good()) {
      if (type[0] == '#') {
         // Ignore comment lines.
         getline(in, type);
      } else if (type.compare("sphere") == 0) {
         Vector center;
         double radius;
         Color color;
         double shininess;
         double reflectivity;

         in >> center.x >> center.y >> center.z;
         in >> radius;
         in >> color.r >> color.g >> color.b;
         in >> shininess;
         in >> reflectivity;

         addObject(new Sphere(center, radius, color, shininess, reflectivity));
      } else if (type.compare("light") == 0) {
         Vector position;
         double intensity;

         in >> position.x >> position.y >> position.z;
         in >> intensity;

         addLight(new Light(position, intensity));
      } else if (type.compare("dispersion") == 0) {
         in >> dispersion;
      } else if (type.compare("maxReflections") == 0) {
         in >> maxReflections;
      } else if (type.compare("cameraUp") == 0) {
         in >> camera.up.x;
         in >> camera.up.y;
         in >> camera.up.z;
      } else if (type.compare("cameraPosition") == 0) {
         in >> camera.position.x;
         in >> camera.position.y;
         in >> camera.position.z;
      } else if (type.compare("cameraLookAt") == 0) {
         in >> camera.lookAt.x;
         in >> camera.lookAt.y;
         in >> camera.lookAt.z;
      } else if (type.compare("imageScale") == 0) {
         in >> imageScale;
      } else {
         cerr << "Type not found: " << type << endl;
         exit(EXIT_FAILURE);
      }

      in >> type;
   }
}


__host__ __device__ Vector Vector::normalize() {
   float len = this->length();
   return (*this) / len;
}

__host__ __device__ Vector Vector::cross(Vector const & v) const {
   return Vector(y*v.z - v.y*z, v.x*z - x*v.z, x*v.y - v.x*y);
}

__device__ double Vector::dot(Vector const & v) const {
   return x*v.x + y*v.y + z*v.z;
}

__host__ __device__ double Vector::length() const {
   return sqrtf(x*x + y*y + z*z);
}

__device__ Vector Vector::operator + (Vector const & v) const {
   return Vector(x+v.x, y+v.y, z+v.z);
}

__device__ Vector & Vector::operator += (Vector const & v) {
   x += v.x;
   y += v.y;
   z += v.z;

   return * this;
}

__host__ __device__ Vector Vector::operator - (Vector const & v) const {
   return Vector(x-v.x, y-v.y, z-v.z);
}

__device__ Vector & Vector::operator -= (Vector const & v) {
   x -= v.x;
   y -= v.y;
   z -= v.z;

   return * this;
}

__device__ Vector Vector::operator * (Vector const & v) const {
   return Vector(x*v.x, y*v.y, z*v.z);
}

__device__ Vector & Vector::operator *= (Vector const & v) {
   x *= v.x;
   y *= v.y;
   z *= v.z;

   return * this;
}

__host__ __device__ Vector Vector::operator / (Vector const & v) const {
   return Vector(x/v.x, y/v.y, z/v.z);
}

__host__ __device__ Vector & Vector::operator /= (Vector const & v) {
   x /= v.x;
   y /= v.y;
   z /= v.z;

   return * this;
}

__device__ Vector Vector::operator * (double const s) const {
   return Vector(x*s, y*s, z*s);
}

__device__ Vector & Vector::operator *= (double const s) {
   x *= s;
   y *= s;
   z *= s;

   return * this;
}

__host__ __device__ Vector Vector::operator / (double const s) const {
   return Vector(x/s, y/s, z/s);
}

__device__ Vector & Vector::operator /= (double const s) {
   x /= s;
   y /= s;
   z /= s;

   return * this;
}

__device__ Intersection Sphere::intersect(Ray ray) {
   Vector deltap = ray.origin - center;
   double a = ray.direction.dot(ray.direction);
   double b = deltap.dot(ray.direction) * 2;
   double c = deltap.dot(deltap) - (radius * radius);

   double disc = b * b - 4 * a * c;
   if (disc < 0) {
      return Intersection(false); // No intersection.
   }

   disc = sqrt(disc);

   double q;
   if (b < 0) {
      q = (-b - disc) * 0.5;
   } else {
      q = (-b + disc) * 0.5;
   }

   double r1 = q / a;
   double r2 = c / q;

   if (r1 > r2) {
      double tmp = r1;
      r1 = r2;
      r2 = tmp;
   }

   double distance = r1;
   if (distance < 0) {
      distance = r2;
   }

   if (distance < 0 || isnan(distance)) {
      return Intersection(false); // No intersection.
   }

   Vector point = ray.origin + (ray.direction * distance);
   Vector normal = (point - center).normalize();

   /* return Intersection(point, distance, normal, Color(fabs(normal.x), fabs(normal.y), fabs(normal.z)), this); */
   return Intersection(ray, point, distance, normal, color, this);
}

__device__ double Sphere::getShininess() {
   return shininess;
}

__device__ double Sphere::getReflectivity() {
   return reflectivity;
}

void Camera::calculateWUV() {
   w = (lookAt - position).normalize();
   u = up.cross(w).normalize();
   v = w.cross(u);
}

__device__ Color Color::operator+ (Color const &c) const {
   Color other;

   other.r = NTZ(c.r) + NTZ(r);
   other.g = NTZ(c.g) + NTZ(g);
   other.b = NTZ(c.b) + NTZ(b);

   return other;
}

__device__ Color Color::operator* (double amount) const {
   Color other;

   other.r = r * amount;
   other.g = g * amount;
   other.b = b * amount;

   return other;
}

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

void Image::setPixmap(Color* pixmap) {
   _pixmap = pixmap;
}
