<?php

GenerateBallsOnAPlane::main();

class GenerateBallsOnAPlane {
   public static function main() {
      echo <<<EOT
cameraUp 0 1 0
cameraPosition 995 400 0
cameraLookAt 100 0 0
cameraScreenWidth 1500

#dispersion 5.0

light  900 900 0  0.7

sphere  1001000        0        0  1000000  FlatColor   1.0000 0.8901 0.8705  null
sphere -1001000        0        0  1000000  FlatColor   0.6941 0.4980 0.4627  null
sphere        0        0  1001000  1000000  FlatColor   0.9921 0.7960 0.7647  null
sphere        0        0 -1001000  1000000  FlatColor   0.3921 0.6941 0.5647  null
sphere        0 -1000000        0  1000000  FlatColor   0.7647 0.9921 0.8941  null
sphere        0  1001000        0  1000000  FlatColor   0.7647 0.9921 0.8941  null

sphere 0 75 0  75  Glass 2.0 50 null


EOT;

      $radius = 0;

      for ($x = -990; $x < 990; $x += $radius + rand(22, 50)) {
         for ($y = -990; $y < 990; $y += $radius + rand(22, 50)) {
            if (abs($x) < 80 && abs($y) < 80) {
               // Leave space for large glass sphere.
               continue;
            }

            $radius = rand(15, 25);

            self::outputSphere(
               $x + rand(-15, 15),
               $y + rand(-15, 15),
               $radius,
               self::getRandomMaterial()
            );
         }
      }
   }

   private static function outputSphere($x, $z, $radius, $material) {
      echo "sphere $x $radius $z  $radius  $material\n";
   }

   private static function getRandomMaterial() {
      $materials = array(
         /* 'FlatColor' => function() { */
         /*    return 'FlatColor ' . GenerateBallsOnAPlane::getRandomColor(); */
         /* }, */
         'ShinyColor' => function() {
            return 'ShinyColor ' . GenerateBallsOnAPlane::getRandomColor() . ' ' .
             GenerateBallsOnAPlane::getRandomSpecularReflective();
         },
         'Checkerboard' => function() {
            return 'Checkerboard ' . GenerateBallsOnAPlane::getRandomColor() . ' ' .
             GenerateBallsOnAPlane::getRandomColor() . ' ' . rand(1, 20) . ' ' .
             GenerateBallsOnAPlane::getRandomSpecularReflective();
         },
         'Glass' => function() {
            return 'Glass ' . (rand(1, 1000) / 200) . ' ' . rand(3, 150);
         },
         'Turbulence' => function() {
            return 'Turbulence ' . GenerateBallsOnAPlane::getRandomColor() . ' ' .
             GenerateBallsOnAPlane::getRandomColor() . ' ' . rand(1, 20) . ' ' .
             GenerateBallsOnAPlane::getRandomSpecularReflective();
         },
         'Marble' => function() {
            return 'Marble ' . GenerateBallsOnAPlane::getRandomColor() . ' ' .
             GenerateBallsOnAPlane::getRandomColor() . ' ' . rand(1, 20) . ' ' .
             GenerateBallsOnAPlane::getRandomSpecularReflective();
         },
         'CrissCross' => function() {
            return 'CrissCross ' . GenerateBallsOnAPlane::getRandomColor() . ' ' .
             GenerateBallsOnAPlane::getRandomColor() . ' ' .
             GenerateBallsOnAPlane::getRandomColor() . ' ' . rand(1, 20) . ' ' .
             GenerateBallsOnAPlane::getRandomSpecularReflective();
         }
      );

      $material = $materials[array_rand($materials)];
      $material = $material();
      $normalMap = self::getRandomNormalMap();

      return "$material $normalMap";
   }

   private static function getRandomNormalMap() {
      $normalMaps = array(
         'NormalMap' => function() {
            return 'NormalMap ' . rand(1, 1000)/50 . ' ' .
             (rand(0, 1000) / 800);
         },
         'null' => function() {
            return 'null';
         }
      );

      $normalMap = $normalMaps[array_rand($normalMaps)];
      return $normalMap();
   }

   public static function getRandomColor() {
      return implode(' ', array(
         'r' => rand(0, 1000)/1000,
         'g' => rand(0, 1000)/1000,
         'b' => rand(0, 1000)/1000
      ));
   }

   public static function getRandomSpecularReflective() {
      return rand(3, 150) . ' ' . (rand(0, 70) / 100.0);
   }
}
