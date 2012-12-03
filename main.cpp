#include <stdio.h>
#include <iostream>
#include <fstream>
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
   int maxReflections;

   vector<Object*> objects;
   vector<Light*> lights;

   RayTracer(int width_, int height_, int maxReflections_) :
    width(width_), height(height_), maxReflections(maxReflections_) {}

   ~RayTracer();

   void addObject(Object* object) {
      objects.push_back(object);
   }

   void addLight(Light* light) {
      lights.push_back(light);
   }

   void traceRays(string);
   Ray getRay(int, int);
   Color castRay(Ray);
   Intersection getClosestIntersection(Ray);
   Color performLighting(Intersection);
   Color getAmbientLighting(Intersection);
   Color getDiffuseAndSpecularLighting(Intersection);
   Color getSpecularLighting(Intersection, Light*);
   Color getReflectiveLighting(Intersection);
   Vector reflectVector(Vector, Vector);
   void readScene(istream&);
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
         image.pixel(x, y, castRay(getRay(x, y)));
      }
   }

   image.WriteTga(fileName.c_str(), true);
}

Ray RayTracer::getRay(int x, int y) {
   int rayX = x - width / 2;
   int rayY = y - height / 2;
   return Ray(Vector(rayX, rayY, 100), Vector(0, 0, -1), maxReflections);
}

Color RayTracer::castRay(Ray ray) {
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

      if (intersection.didIntersect && intersection.distance <
       closestIntersection.distance) {
         closestIntersection = intersection;
      }
   }

   return closestIntersection;
}

Color RayTracer::performLighting(Intersection intersection) {
   Color ambientColor = getAmbientLighting(intersection);
   Color diffuseAndSpecularColor = getDiffuseAndSpecularLighting(intersection);
   Color reflectedColor = getReflectiveLighting(intersection);

   return ambientColor + diffuseAndSpecularColor + reflectedColor;
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
         Ray shadowRay = Ray(intersection.intersection, lightDirection, 1);
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
   Vector reflected = reflectVector(lightOffset.normalize(), intersection.normal);

   double dot = view.dot(reflected);

   if (dot <= 0) {
      return specularColor;
   }

   double specularAmount = pow(dot, shininess);

   specularColor.r = specularAmount;
   specularColor.g = specularAmount;
   specularColor.b = specularAmount;

   return specularColor;
}

Color RayTracer::getReflectiveLighting(Intersection intersection) {
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

Vector RayTracer::reflectVector(Vector vector, Vector normal) {
   return normal * 2 * vector.dot(normal) - vector;
}

void RayTracer::readScene(istream& in) {
   string type;

   while (in.good()) {
      in >> type;

      if (type.compare("sphere") == 0) {
         cout << "sphere!" << endl;
      } else if (type.compare("light") == 0) {
         cout << "light!" << endl;
      } else {
         cout << "Nope" << endl;
      }
   }
}

/**
 * RayTracer main.
 */
int main(int argc, char** argv) {
   if (argc < 2) {
      cerr << "No scene file provided!" << endl;
      cerr << "Usage: " << argv[0] << " sceneFile [outFile]" << endl;
      exit(EXIT_FAILURE);
   }

   RayTracer rayTracer(600, 600, 10);

   if (strcmp(argv[1], "-") == 0) {
      rayTracer.readScene(cin);
   } else {
      char* inFile = argv[1];
      ifstream inFileStream;
      inFileStream.open(inFile, ifstream::in);

      if (inFileStream.fail()) {
         cerr << "Failed opening file" << endl;
         exit(EXIT_FAILURE);
      }

      rayTracer.readScene(inFileStream);
      inFileStream.close();
   }

   string outFile;
   if (argc > 2) {
      outFile = argv[2];
   } else {
      cerr << "No outFile specified - writing to out.tga" << endl;
      outFile = "out.tga";
   }

   /* Two spheres with a shadow. */
   /* rayTracer.addObject( */
   /*  new Sphere(Vector(-150, 0, -150), 150, Color(1.0, 0.0, 0.0), 10, 0.5)); */
   /* rayTracer.addObject( */
   /*  new Sphere(Vector(50, 50, 25), 25, Color(0.0, 1.0, 0.0), 10, 0.5)); */

   /* Two spheres next to each other for reflections. */
   /* rayTracer.addObject( */
   /*  new Sphere(Vector(-105, -75, -150), 100, Color(1.0, 0.0, 0.0), 100, 0.5)); */
   /* rayTracer.addObject( */
   /*  new Sphere(Vector(105, -75, -150), 100, Color(0.0, 1.0, 0.0), 5, 0.8)); */
   /* rayTracer.addObject( */
   /*  new Sphere(Vector(0, 100, -150), 100, Color(0.0, 0.0, 1.0), 100, 0.5)); */

   rayTracer.addLight(new Light(Vector(300, 100, 150)));
   rayTracer.addLight(new Light(Vector(-300, 100, 150)));

   rayTracer.traceRays(outFile);

   exit(EXIT_SUCCESS);
}
