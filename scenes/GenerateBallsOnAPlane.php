<?php

GenerateBallsOnAPlane::main();

class GenerateBallsOnAPlane {
   public static function main() {
      echo <<<EOT
cameraUp 0 1 0
cameraPosition 950 300 950
cameraLookAt 300 0 0
cameraScreenWidth 1500

#dispersion 5.0

light  900 900  900  0.6

sphere  1001000        0        0  1000000  FlatColor  1 0 0  null
sphere -1001000        0        0  1000000  FlatColor  1 1 0  null
sphere        0        0  1001000  1000000  FlatColor  0 1 0  null
sphere        0        0 -1001000  1000000  FlatColor  0 1 1  null
sphere        0 -1000000        0  1000000  FlatColor  1 1 1  null
sphere        0  1001000        0  1000000  FlatColor  1 0 1  null


EOT;

      $radius = 0;

      for ($x = -1000; $x < 1000; $x += $radius + rand(10, 50)) {
         for ($y = -1000; $y < 1000; $y += $radius + rand(10, 50)) {
            $radius = rand(8, 45);

            self::outputSphere(
               $x + rand(-5, 5),
               $y + rand(-5, 5),
               rand(8, 24),
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
         'FlatColor' => function() {
            return 'FlatColor ' . GenerateBallsOnAPlane::getRandomColor();
         },
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
