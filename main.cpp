#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <limits>
#include "Image.h"
#include "Object.h"
#include "Sphere.h"
#include "Intersection.h"
#include "Light.h"

using namespace std;

class RayTracer {
public:
   int width;
   int height;

   vector<Object*> objects;
   vector<Light*> lights;

   RayTracer(int width_, int height_) : width(width_), height(height_) {}

   ~RayTracer();

   void addObject(Object* object) {
      objects.push_back(object);
   }

   void addLight(Light* light) {
      lights.push_back(light);
   }

   void traceRays(string);
   Color castRay(int, int);
   Intersection getClosestIntersection(Ray);
   Color performLighting(Intersection);
   Color getAmbientLighting(Intersection);
   Color getDiffuseAndSpecularLighting(Intersection);
   Color getSpecularLighting(Intersection, Light*);
};

RayTracer::~RayTracer() {
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      delete *itr;
   }

   for (vector<Light*>::iterator itr = lights.begin(); itr < lights.end(); itr++) {
      delete *itr;
   }
}

void RayTracer::traceRays(string fileName) {
   Image image(width, height);

   for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
         image.pixel(x, y, castRay(x, y));
      }
   }

   image.WriteTga(fileName.c_str(), true);
}

Color RayTracer::castRay(int x, int y) {
   int rayX = x - width / 2;
   int rayY = y - height / 2;
   Ray ray(Vector(rayX, rayY, 100), Vector(0, 0, -1));

   Intersection intersection = getClosestIntersection(ray);

   if (intersection.didIntersect) {
      return performLighting(intersection);
   } else {
      return Color();
   }
}

Intersection RayTracer::getClosestIntersection(Ray ray) {
   Intersection closestIntersection(false);
   closestIntersection.distance = numeric_limits<double>::max();

   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect && intersection.distance < closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection;
}

Color RayTracer::performLighting(Intersection intersection) {
   Color ambientColor = getAmbientLighting(intersection);
   Color diffuseAndSpecularColor = getDiffuseAndSpecularLighting(intersection);

   return ambientColor + diffuseAndSpecularColor;
}

Color RayTracer::getAmbientLighting(Intersection intersection) {
   return intersection.color * 0.2;
}

Color RayTracer::getDiffuseAndSpecularLighting(Intersection intersection) {
   Color diffuseColor(0.0, 0.0, 0.0);
   Color specularColor(0.0, 0.0, 0.0);

   for (vector<Light*>::iterator itr = lights.begin(); itr < lights.end(); itr++) {
      Light* light = *itr;
      Vector lightOffset = light->position - intersection.intersection;
      /**
       * TODO: Be careful about normalizing lightOffset too.
       */
      Vector lightDirection = lightOffset.normalize();
      double dotProduct = intersection.normal.dot(lightDirection);

      /**
       * Intersection is facing light.
       */
      if (dotProduct >= 0.0f) {
         Ray shadowRay = Ray(intersection.intersection + lightDirection, lightDirection);
         Intersection shadowIntersection = getClosestIntersection(shadowRay);

         if (shadowIntersection.didIntersect) {
            /**
             * Position is in shadow of another object - continue with other lights.
             */
            continue;
         }

         diffuseColor = diffuseColor + (intersection.color * dotProduct);
         specularColor = specularColor + getSpecularLighting(intersection, light);
      }
   }

   return diffuseColor + specularColor;
}

Color RayTracer::getSpecularLighting(Intersection intersection, Light* light) {
   Color specularColor(0.0, 0.0, 0.0);
   double shininess = intersection.object->getShininess();

   if (shininess == NOT_SHINY) {
      /* Don't perform specular lighting on non shiny objects. */
      return specularColor;
   }

   Vector view = (intersection.ray.origin - intersection.intersection).normalize();
   Vector lightOffset = light->position - intersection.intersection;
   Vector L = lightOffset.normalize();
   Vector N = intersection.normal;

   /* R = -L + 2(L dot N)N = 2 * N * (L dot N) - L */
   Vector R = N * 2 * L.dot(N) - L;

   double dot = view.dot(R);

   if (dot <= 0) {
      return specularColor;
   }

   double specularAmount = pow(dot, shininess);

   specularColor.r = specularAmount;
   specularColor.g = specularAmount;
   specularColor.b = specularAmount;

   return specularColor;
}

/**
 * RayTracer main.
 */
int main(void) {
   RayTracer rayTracer(600, 600);
   string fileName = "awesome.tga";

   rayTracer.addObject(
    new Sphere(Vector(-150, 0, -150), 150, Color(1.0, 0.0, 0.0), 10, 0.5));
   rayTracer.addObject(
    new Sphere(Vector(50, 50, 25), 25, Color(0.0, 1.0, 0.0), 10, 0.5));

   rayTracer.addLight(new Light(Vector(300, 100, 150)));
   rayTracer.addLight(new Light(Vector(-300, 100, 150)));

   rayTracer.traceRays(fileName);

   return 0;
}
