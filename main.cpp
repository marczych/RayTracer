#include <iostream>
#include <fstream>
#include "RayTracer.h"

using namespace std;

/**
 * RayTracer main.
 */
int main(int argc, char** argv) {
   if (argc < 2) {
      cerr << "No scene file provided!" << endl;
      cerr << "Usage: " << argv[0] << " sceneFile [outFile]" << endl;
      exit(EXIT_FAILURE);
   }

   RayTracer rayTracer(1024, 768, 10);

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

   rayTracer.traceRays(outFile);

   exit(EXIT_SUCCESS);
}
