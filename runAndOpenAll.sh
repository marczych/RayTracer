#!/bin/bash

# Runs RayTracer on every scene file in scenes and opens the resulting images.

make

scenes=`ls scenes`
for scene in $scenes
do
   outFile=`echo $scene | sed 's/\..*/.tga'/`

   ./RayTracer scenes/$scene $outFile
   open $outFile
done

