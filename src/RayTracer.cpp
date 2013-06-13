#include <limits>
#include "RayTracer.h"
#include "Image.h"
#include "Object.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Intersection.h"
#include "Light.h"
#include "Air.h"
#include "ShinyColor.h"
#include "FlatColor.h"
#include "Checkerboard.h"
#include "Marble.h"
#include "Wood.h"
#include "Glass.h"
#include "NormalMap.h"
#include "Turbulence.h"
#include "CrissCross.h"
#include "BSP.h"

using namespace std;

RayTracer::RayTracer(int width_, int height_, int maxReflections_, int superSamples_,
 int depthComplexity_) : width(width_), height(height_),
 maxReflections(maxReflections_), superSamples(superSamples_), camera(Camera()),
 imageScale(1), depthComplexity(depthComplexity_), dispersion(5.0f), raysCast(0),
 startingMaterial(new Air()) {}

RayTracer::~RayTracer() {
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      delete *itr;
   }

   for (vector<Light*>::iterator itr = lights.begin(); itr < lights.end(); itr++) {
      delete *itr;
   }

   delete startingMaterial;
}

void RayTracer::traceRays(string fileName) {
   int columnsCompleted = 0;
   camera.calculateWUV();
   Image image(width, height);

   // Reset depthComplexity to avoid unnecessary loops.
   if (dispersion < 0) {
      depthComplexity = 1;
   }

   imageScale = camera.screenWidth / (float)width;

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

Color RayTracer::castRayAtPoint(const Vector& point) {
   Color color;

   for (int i = 0; i < depthComplexity; i++) {
      Ray viewRay(camera.position, point - camera.position, maxReflections,
       startingMaterial);

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

Color RayTracer::castRay(const Ray& ray) {
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
bool RayTracer::isInShadow(const Ray& ray, double lightDistance) {
   for (vector<Object*>::iterator itr = objects.begin(); itr < objects.end(); itr++) {
      Intersection intersection = (*itr)->intersect(ray);

      if (intersection.didIntersect && intersection.distance < lightDistance) {
         return true;
      }
   }

   return false;
}

Intersection RayTracer::getClosestIntersection(const Ray& ray) {
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

Color RayTracer::performLighting(const Intersection& intersection) {
   Color color = intersection.getColor();
   Color ambientColor = getAmbientLighting(intersection, color);
   Color diffuseAndSpecularColor = getDiffuseAndSpecularLighting(intersection, color);
   Color reflectedColor = getReflectiveRefractiveLighting(intersection);

   return ambientColor + diffuseAndSpecularColor + reflectedColor;
}

Color RayTracer::getAmbientLighting(const Intersection& intersection, const Color& color) {
   return color * 0.2;
}

Color RayTracer::getDiffuseAndSpecularLighting(const Intersection& intersection,
 const Color& color) {
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
         Ray shadowRay = Ray(intersection.intersection, lightDirection, 1,
          intersection.ray.material);

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

Color RayTracer::getSpecularLighting(const Intersection& intersection,
 Light* light) {
   Color specularColor(0.0, 0.0, 0.0);
   double shininess = intersection.endMaterial->getShininess();

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

Color RayTracer::getReflectiveRefractiveLighting(const Intersection& intersection) {
   double reflectivity = intersection.endMaterial->getReflectivity();
   double startRefractiveIndex = intersection.startMaterial->getRefractiveIndex();
   double endRefractiveIndex = intersection.endMaterial->getRefractiveIndex();
   int reflectionsRemaining = intersection.ray.reflectionsRemaining;

   /**
    * Don't perform lighting if the object is not reflective or refractive or we have
    * hit our recursion limit.
    */
   if (reflectivity == NOT_REFLECTIVE && endRefractiveIndex == NOT_REFRACTIVE ||
    reflectionsRemaining <= 0) {
      return Color();
   }

   // Default to exclusively reflective values.
   double reflectivePercentage = reflectivity;
   double refractivePercentage = 0;

   // Refractive index overrides the reflective property.
   if (endRefractiveIndex != NOT_REFRACTIVE) {
      reflectivePercentage = getReflectance(intersection.normal,
       intersection.ray.direction, startRefractiveIndex, endRefractiveIndex);

      refractivePercentage = 1 - reflectivePercentage;
   }

   // No ref{ra,le}ctive properties - bail early.
   if (refractivePercentage <= 0 && reflectivePercentage <= 0) {
      return Color();
   }

   Color reflectiveColor;
   Color refractiveColor;

   if (reflectivePercentage > 0) {
      Vector reflected = reflectVector(intersection.ray.origin,
       intersection.normal);
      Ray reflectedRay(intersection.intersection, reflected, reflectionsRemaining - 1,
       intersection.ray.material);

      reflectiveColor = castRay(reflectedRay) * reflectivePercentage;
   }

   if (refractivePercentage > 0) {
      Vector refracted = refractVector(intersection.normal,
       intersection.ray.direction, startRefractiveIndex, endRefractiveIndex);
      Ray refractedRay = Ray(intersection.intersection, refracted, 1,
       intersection.endMaterial);

      refractiveColor = castRay(refractedRay) * refractivePercentage;
   }

   return reflectiveColor + refractiveColor;
}

double RayTracer::getReflectance(const Vector& normal, const Vector& incident,
 double n1, double n2) {
   double n = n1 / n2;
   double cosI = -normal.dot(incident);
   double sinT2 = n * n * (1.0 - cosI * cosI);

   if (sinT2 > 1.0) {
      // Total Internal Reflection.
      return 1.0;
   }

   double cosT = sqrt(1.0 - sinT2);
   double r0rth = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
   double rPar = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT);
   return (r0rth * r0rth + rPar * rPar) / 2.0;
}

Vector RayTracer::refractVector(const Vector& normal, const Vector& incident,
 double n1, double n2) {
   double n = n1 / n2;
   double cosI = -normal.dot(incident);
   double sinT2 = n * n * (1.0 - cosI * cosI);

   if (sinT2 > 1.0) {
      cerr << "Bad refraction vector!" << endl;
      exit(EXIT_FAILURE);
   }

   double cosT = sqrt(1.0 - sinT2);
   return incident * n + normal * (n * cosI - cosT);
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
      } else if (type.compare("model") == 0) {
         string model;
         int size;
         Vector translate;
         Material* material;

         in >> model;
         in >> size;
         in >> translate.x >> translate.y >> translate.z;

         material = readMaterial(in);
         readModel(model, size, translate, material);
      } else if (type.compare("material") == 0) {
         addMaterial(in);
      } else if (type.compare("triangle") == 0) {
         Vector v0, v1, v2;
         Material* material;

         in >> v0.x >> v0.y >> v0.z;
         in >> v1.x >> v1.y >> v1.z;
         in >> v2.x >> v2.y >> v2.z;
         material = readMaterial(in);

         addObject(new Triangle(v0, v1, v2, material));
      } else if (type.compare("sphere") == 0) {
         Vector center;
         double radius;
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
      } else if (type.compare("startingMaterial") == 0) {
         startingMaterial = readMaterial(in);
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
      } else if (type.compare("cameraScreenWidth") == 0) {
         in >> camera.screenWidth;
      } else {
         cerr << "Type not found: " << type << endl;
         exit(EXIT_FAILURE);
      }

      in >> type;
   }

   bsp = new BSP(0, 'x', objects);
   cout << "\nDONE\n";
}

void RayTracer::readModel(string model, int size, Vector translate, Material* material) {
   string type;
   vector<Vector> vertices;
   Vector centerOffset;
   double minX, maxX, minY, maxY, minZ, maxZ;
   double offX = 0.0, offY = 0.0, offZ = 0.0, scale = 0.0;

   cout << model;

   ifstream in;
   in.open(model.c_str(), ifstream::in);

   if (in.fail()) {
      cerr << "Failed opening file" << endl;
      exit(EXIT_FAILURE);
   }

   in >> type;
   while (in.good()) {
      if (type.compare("Vertex") == 0) {
         int index;
         Vector v;

         in >> index;
         in >> v.x >> v.y >> v.z;

         minX = min(minX, v.x);
         minY = min(minY, v.y);
         minZ = min(minZ, v.z);

         maxX = max(maxX, v.x);
         maxY = max(maxY, v.y);
         maxZ = max(maxZ, v.z);

         vertices.push_back(v);
      } else if (type.compare("Face") == 0) {
         if (scale == 0.0) {
            offX = (maxX + minX) / 2;
            offY = (maxY + minY) / 2;
            offZ = (maxZ + minZ) / 2;
            centerOffset = Vector(offX, offY, offZ);

            double distance = sqrt(pow((maxX - minX), 2) +
                                   pow((maxY - minY), 2) +
                                   pow((maxZ - minZ), 2));

            if (distance == 0.0)
               exit(EXIT_FAILURE);

            scale = size / distance;

            printf("\n\nMin X: %f, Y: %f, Z: %f\n", minX, minY, minZ);
            printf("Max X: %f, Y: %f, Z: %f\n", maxX, maxY, maxZ);
            printf("Diff (%f, %f, %f) Scale %f\n", offX, offY, offZ, scale);
            printf("Distance: %f\n", distance);
         }

         int face, v0, v1, v2;

         in >> face >> v0 >> v1 >> v2;

         Vector a = (vertices[v0 - 1] - centerOffset) * scale + translate;
         Vector b = (vertices[v1 - 1] - centerOffset) * scale + translate;
         Vector c = (vertices[v2 - 1] - centerOffset) * scale + translate;

         addObject(new Triangle(a, b, c, material));
      }

      in >> type;
   }

   in.close();
}



/**
 * Parses the input stream and makes a new Material.
 */
Material* RayTracer::readMaterial(istream& in) {
   Material* material;
   string type;
   in >> type;

   if (type.compare("FlatColor") == 0) {
      material = new FlatColor(in);
   } else if (type.compare("ShinyColor") == 0) {
      material = new ShinyColor(in);
   } else if (type.compare("Checkerboard") == 0) {
      material = new Checkerboard(in);
   } else if (type.compare("Glass") == 0) {
      material = new Glass(in);
   } else if (type.compare("Turbulence") == 0) {
      material = new Turbulence(in);
   } else if (type.compare("Marble") == 0) {
      material = new Marble(in);
   } else if (type.compare("CrissCross") == 0) {
      material = new CrissCross(in);
   } else if (type.compare("Wood") == 0) {
      material = new Wood(in);
   } else if (materials.count(type) > 0) {
      material = materials[type];

      // Stored material already has the NormalMap so return here to avoid
      // scene parsing problems below.
      return material;
   } else {
      cerr << "Material not found: " << type << endl;
      exit(EXIT_FAILURE);
   }

   material->setNormalMap(readNormalMap(in));

   return material;
}

NormalMap* RayTracer::readNormalMap(istream& in) {
   string type;
   in >> type;

   if (type.compare("null") == 0) {
      return NULL;
   } else if (type.compare("NormalMap") == 0) {
      return new NormalMap(in);
   } else {
      cerr << "NormalMap not found: " << type << endl;
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
