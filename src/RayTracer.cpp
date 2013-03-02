#include <limits>
#include "RayTracer.h"
#include "Image.h"
#include "Object.h"
#include "Sphere.h"
#include "Intersection.h"
#include "Light.h"
#include "FlatColor.h"
#include "Checkerboard.h"

using namespace std;

RayTracer::RayTracer(int width_, int height_, int maxReflections_, int superSamples_,
 int depthComplexity_) : width(width_), height(height_),
 maxReflections(maxReflections_), superSamples(superSamples_), camera(Camera()),
 imageScale(1), depthComplexity(depthComplexity_), dispersion(5.0f), raysCast(0) {}

RayTracer::~RayTracer() {
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      delete *itr;
   }

   for (vector<Light*>::iterator itr = lights.begin(); itr < lights.end(); itr++) {
      delete *itr;
   }
}

void RayTracer::traceRays(string fileName) {
   int columnsCompleted = 0;
   Image image(width, height);

   // Reset depthComplexity to avoid unnecessary loops.
   if (dispersion < 0) {
      depthComplexity = 1;
   }

   #pragma omp parallel for
   for (int x = 0; x < width; x++) {
      // Update percent complete.
      columnsCompleted++;
      float percentage = columnsCompleted/(float)width * 100;
      cout << '\r' << (int)percentage << '%';
      fflush(stdout);

      for (int y = 0; y < height; y++) {
         image.pixel(x, y, castRayForPixel(x, y));
      }
   }

   cout << "\rDone!" << endl;
   cout << "Rays cast: " << raysCast << endl;

   image.WriteTga(fileName.c_str(), false);
}

Color RayTracer::castRayForPixel(int x, int y) {
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

         color = color + (castRayAtPoint(imagePlanePoint) * sampleWeight);
      }
   }

   return color;
}

Color RayTracer::castRayAtPoint(Vector point) {
   Color color;

   for (int i = 0; i < depthComplexity; i++) {
      Ray viewRay(camera.position, point - camera.position, maxReflections);

      if (depthComplexity > 1) {
         Vector disturbance(
          (dispersion / RAND_MAX) * (1.0f * rand()),
          (dispersion / RAND_MAX) * (1.0f * rand()),
          0.0f);

         viewRay.origin = viewRay.origin + disturbance;
         viewRay.direction = point - viewRay.origin;
         viewRay.direction = viewRay.direction.normalize();
      }

      color = color + (castRay(viewRay) * (1 / (float)depthComplexity));
   }

   return color;
}

Color RayTracer::castRay(Ray ray) {
   raysCast++;
   Intersection intersection = getClosestIntersection(ray);

   if (intersection.didIntersect) {
      return performLighting(intersection);
   } else {
      return Color();
   }
}

/**
 * Basically same code as getClosestIntersection but short circuits if an
 * intersection closer to the given light distance is found.
 */
bool RayTracer::isInShadow(Ray ray, double lightDistance) {
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect && intersection.distance < lightDistance) {
         return true;
      }
   }

   return false;
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
   Color color = intersection.getColor();
   Color ambientColor = getAmbientLighting(intersection, color);
   Color diffuseAndSpecularColor = getDiffuseAndSpecularLighting(intersection, color);
   Color reflectedColor = getReflectiveLighting(intersection);

   return ambientColor + diffuseAndSpecularColor + reflectedColor;
}

Color RayTracer::getAmbientLighting(Intersection intersection, Color color) {
   return color * 0.2;
}

Color RayTracer::getDiffuseAndSpecularLighting(Intersection intersection, Color color) {
   Color diffuseColor(0.0, 0.0, 0.0);
   Color specularColor(0.0, 0.0, 0.0);

   for (vector<Light*>::iterator itr = lights.begin(); itr < lights.end(); itr++) {
      Light* light = *itr;
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

         if (isInShadow(shadowRay, lightDistance)) {
            /**
             * Position is in shadow of another object - continue with other lights.
             */
            continue;
         }

         diffuseColor = (diffuseColor + (color * dotProduct)) *
          light->intensity;
         specularColor = specularColor + getSpecularLighting(intersection, light);
      }
   }

   return diffuseColor + specularColor;
}

Color RayTracer::getSpecularLighting(Intersection intersection, Light* light) {
   Color specularColor(0.0, 0.0, 0.0);
   double shininess = intersection.material->getShininess();

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

Color RayTracer::getReflectiveLighting(Intersection intersection) {
   double reflectivity = intersection.material->getReflectivity();
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

   in >> type;

   while (in.good()) {
      if (type[0] == '#') {
         // Ignore comment lines.
         getline(in, type);
      } else if (type.compare("material") == 0) {
         addMaterial(in);
      } else if (type.compare("sphere") == 0) {
         Vector center;
         double radius;
         Color color;
         Material* material;

         in >> center.x >> center.y >> center.z;
         in >> radius;
         material = readMaterial(in);

         addObject(new Sphere(center, radius, material));
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

/**
 * Parses the input stream and makes a new Material.
 */
Material* RayTracer::readMaterial(istream& in) {
   string type;
   in >> type;

   if (type.compare("FlatColor") == 0) {
      FlatColor* material = new FlatColor();

      in >> material->color.r >> material->color.g >> material->color.b;
      in >> material->shininess;
      in >> material->reflectivity;

      return material;
   } else if (type.compare("Checkerboard") == 0) {
      Checkerboard* material = new Checkerboard();

      in >> material->color1.r >> material->color1.g >> material->color1.b;
      in >> material->color2.r >> material->color2.g >> material->color2.b;
      in >> material->scale;
      in >> material->shininess;
      in >> material->reflectivity;

      return material;
   } else if (materials.count(type) > 0) {
      return materials[type];
   } else {
      cerr << "Type not found: " << type << endl;
      exit(EXIT_FAILURE);
   }
}

void RayTracer::addMaterial(istream& in) {
   string materialName;

   in >> materialName;

   for (string::iterator itr = materialName.begin(); itr < materialName.end(); itr++) {
      if (isupper(*itr)) {
         cerr << "Invalid material name: " << materialName << endl;
         exit(EXIT_FAILURE);
      }
   }

   if (materials.count(materialName) > 0) {
      cerr << "Duplicate material name: " << materialName << endl;
      exit(EXIT_FAILURE);
   }

   Material* material = readMaterial(in);
   materials.insert(pair<string, Material*>(materialName, material));
}
