<?php

GenerateBallsOnAPlane::main();

class GenerateBallsOnAPlane {
   public static function main() {
      echo <<<EOT
dispersion 5.0
focus 100
cameraPosition 250
sphere 0 0 -10000  10000  0.7 0.3 0.4  -1  0.4
light 0 0 40  0.9

EOT;

      $radius = 0;

      for ($x = -150; $x < 150; $x += $radius + rand(4, 15)) {
         for ($y = -150; $y < 150; $y += $radius + rand(4, 15)) {
            $radius = rand(3, 15);

            self::outputSphere(
               $x + rand(-5, 5),
               $y + rand(-5, 5),
               rand(3, 7),
               self::getRandomColor(),
               rand(3, 150),
               rand(0, 70) / 100.0
            );
         }
      }
   }

   private static function getRandomColor() {
      return array(
         'r' => rand(0, 1000)/1000,
         'g' => rand(0, 1000)/1000,
         'b' => rand(0, 1000)/1000
      );
   }

   private static function outputSphere($x, $y, $radius, $color, $specular,
    $reflectivity) {
      echo "sphere $x $y $radius  $radius  {$color['r']} {$color['g']} {$color['b']} ";
      echo " $specular $reflectivity\n";
   }
}
