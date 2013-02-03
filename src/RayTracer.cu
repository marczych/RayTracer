#include <limits>
#include "RayTracer.h"
#include "Image.h"
#include "Object.h"
#include "Sphere.h"
#include "Intersection.h"
#include "Light.h"

using namespace std;

#define ERROR_HANDLER(x) ErrorHandler(x, __FILE__, __LINE__)
#define TILE_WIDTH 32

static void ErrorHandler(cudaError_t err, const char *file, int line) {
   if (err != cudaSuccess) {
      fprintf(stderr, "%s in line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}

RayTracer::RayTracer(int width_, int height_, int maxReflections_, int superSamples_,
 int depthComplexity_) : width(width_), height(height_),
 maxReflections(maxReflections_), superSamples(superSamples_), camera(Camera()),
 imageScale(1), depthComplexity(depthComplexity_), dispersion(5.0f), raysCast(0) {}

RayTracer::~RayTracer() {
}

void RayTracer::traceRays(string fileName) {
   Color* devImage;
   RayTracer* devRayTracer;
   int columnsCompleted = 0;
   Image image(width, height);

   // Reset depthComplexity to avoid unnecessary loops.
   if (dispersion < 0) {
      depthComplexity = 1;
   }

   ERROR_HANDLER(cudaMalloc((void**)&conSpheres, spheres.size() * sizeof(Sphere)));
   ERROR_HANDLER(cudaMemcpy(conSpheres, &spheres.front(),
    spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));

   ERROR_HANDLER(cudaMalloc((void**)&conLights, lights.size() * sizeof(Light)));
   ERROR_HANDLER(cudaMemcpy(conLights, &lights.front(),
    lights.size() * sizeof(Light), cudaMemcpyHostToDevice));

   ERROR_HANDLER(cudaMalloc((void**)&devImage, width * height * sizeof(Color)));

   ERROR_HANDLER(cudaMalloc((void**)&devRayTracer, sizeof(RayTracer)));
   ERROR_HANDLER(cudaMemcpy(devRayTracer, this, sizeof(RayTracer), cudaMemcpyHostToDevice));

   int gridWidth = ceil((float)width/TILE_WIDTH);
   int gridHeight = ceil((float)height/TILE_WIDTH);

   dim3 dimGrid(gridWidth, gridHeight);
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

   cudaTraceRays<<<dimGrid, dimBlock>>>(devImage, devRayTracer);

   ERROR_HANDLER(cudaMemcpy(image.getPixmap(), devImage,
    width * height * sizeof(Color), cudaMemcpyDeviceToHost));

   ERROR_HANDLER(cudaFree(devImage));
   ERROR_HANDLER(cudaFree(conSpheres));
   ERROR_HANDLER(cudaFree(conLights));
   ERROR_HANDLER(cudaFree(devRayTracer));

   image.WriteTga(fileName.c_str(), false);
}

__global__ void cudaTraceRays(Color* image, RayTracer* rayTracer) {
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
   int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

   if (x < rayTracer->width && y < rayTracer->height) {
      Color color = rayTracer->castRayForPixel(x, y);
      //Color* current
      //image.pixel(x, y, castRayForPixel(x, y));
   }
}

__device__ Color RayTracer::castRayForPixel(int x, int y) {
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
         //Vector imagePlanePoint = camera.lookAt -
         // (camera.u * (sampleStartX + (x * sampleWidth)) * imageScale) +
         // (camera.v * (sampleStartY + (y * sampleWidth)) * imageScale);

         //color = color + (castRayAtPoint(imagePlanePoint) * sampleWeight);
      }
   }

   return color;
}

__device__ Color RayTracer::castRayAtPoint(Vector point) {
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

      color = color + (castRay(viewRay) * (1 / (float)depthComplexity));
   }

   return color;
}

__device__ Color RayTracer::castRay(Ray ray) {
   raysCast++;
   Intersection intersection = getClosestIntersection(ray);

   if (intersection.didIntersect) {
      return performLighting(intersection);
   } else {
      return Color();
   }
}

__device__ Intersection RayTracer::getClosestIntersection(Ray ray) {
   Intersection closestIntersection(false);
   closestIntersection.distance = 983487438;
   int numSpheres = 20; // TODO

   for (int i = 0; i < numSpheres; i++) {
      Sphere* sphere = conSpheres + i;
      Intersection intersection = sphere->intersect(ray);

      if (intersection.didIntersect && intersection.distance <
       closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection;
}

__device__ Color RayTracer::performLighting(Intersection intersection) {
   Color ambientColor = getAmbientLighting(intersection);
   Color diffuseAndSpecularColor = getDiffuseAndSpecularLighting(intersection);
   //Color reflectedColor = getReflectiveLighting(intersection);

   return ambientColor + diffuseAndSpecularColor;// + reflectedColor;
}

__device__ Color RayTracer::getAmbientLighting(Intersection intersection) {
   return intersection.color * 0.2;
}

__device__ Color RayTracer::getDiffuseAndSpecularLighting(Intersection intersection) {
   Color diffuseColor(0.0, 0.0, 0.0);
   Color specularColor(0.0, 0.0, 0.0);
   int numLights = 1; // TODO

   for (int i = 0; i < numLights; i++) {
      Light* light = conLights + i;
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
         Intersection shadowIntersection = getClosestIntersection(shadowRay);

         if (shadowIntersection.didIntersect &&
          shadowIntersection.distance < lightDistance) {
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

__device__ Color RayTracer::getReflectiveLighting(Intersection intersection) {
   double reflectivity = intersection.object->getReflectivity();
   int reflectionsRemaining = intersection.ray.reflectionsRemaining;

   if (reflectivity == NOT_REFLECTIVE || reflectionsRemaining <= 0) {
      return Color();
   } else {
      Vector reflected = reflectVector(intersection.ray.origin, intersection.normal);
      Ray reflectedRay(intersection.intersection, reflected, reflectionsRemaining - 1);

      return castRay(reflectedRay) * reflectivity;
   }
}

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

