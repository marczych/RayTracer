#!/bin/bash

# Runs RayTracer on every scene file in scenes and opens the resulting images.

make

SUPER_SAMPLES=1
DEPTH_COMPLEXITY=5

cd ../scenes
scenes=`ls *.scn`
for scene in $scenes
do
   sceneName=`echo $scene | sed 's/\..*//'`
   outFile=$sceneName
   outFile+="_$SUPER_SAMPLES"
   outFile+="_$DEPTH_COMPLEXITY"
   outFile+=".tga"

   ../src/RayTracer $scene $SUPER_SAMPLES $DEPTH_COMPLEXITY $outFile
   echo $outFile # So you can easily get output files in a list.
done

