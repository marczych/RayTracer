#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include "RayTracer.h"

using namespace std;

/**
 * RayTracer main.
 */
int main(int argc, char** argv) {
   if (argc < 4) {
      cerr << "Usage: " << argv[0] << " sceneFile superSamples " <<
       "depthComplexity [outFile]" << endl;
      exit(EXIT_FAILURE);
   }

   srand((unsigned)time(0));
   int maxReflections = 10;
   int superSamples = atoi(argv[2]);
   int depthComplexity = atoi(argv[3]);
   RayTracer rayTracer(1920, 1080, maxReflections, superSamples, depthComplexity);

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

   if (argc > 4) {
      outFile = argv[4];
   } else {
      cerr << "No outFile specified - writing to out.tga" << endl;
      outFile = "out.tga";
   }

   rayTracer.traceRays(outFile);

   exit(EXIT_SUCCESS);
}
