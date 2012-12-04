#Ray Tracer!

This is a simple ray tracer developed as a final project for CSC 471 Introduction to Graphics at Cal Poly.
It currently supports phong lighting, shadows, reflections, super sampling, and depth of field.
The only objects supported are spheres.

## Basic Ray Tracing
The basic ray tracing model is as follows.
A camera is placed in the world and rays are cast from the camera's position to points on an imaginary image plane.
A ray is used to determine what light would be going towards the camera at that point and direction.
In our case each ray represents a pixel in the output image and the resulting color of the ray determines the color output for the pixel.

## Intersections
Each ray performs an intersection test with all objects in the scene to determine the closest intersection.
The closest intersection is used to determine the resulting color.
[intersection.tga]

## Lighting
In addition to objects, lights are positioned in the world to shade the objects.

### Diffuse
Diffuse lighting is determined by computing the intensity of the light at a point on the sphere.
If the angle is close to the normal at that point then the intensity will be increased.
The intensity determines how much of the object's color to contribute.
[diffuse.tga]

### Shadows
Shadows are incorporated into lighting.
To determine if a light source should contribute to the lighting at an intersection point a shadow ray is cast from the intersection point to the light source.
If there is an intersection before the light source then this point is in the shadow of that light source.
[shadows.tga]

### Specular
Specular lighting is calculated by computing a reflection ray by reflecting the light vector about the normal at the intersection point.
The view ray is compared to the reflection ray to determine how much specular lighting to contribute.
The more parallel the vectors are the more specular lighting will be added.
[specular.tga]

## Super Sampling (Anti-aliasing)
Aliasing is an artifact of images that have sharp, jagged edges.
Super sampling is an anti-aliasing technique to smooth out jagged edges.
My super sampling algorithm works by casting more initial rays and averaging neighboring samples together.
For example, 2x super sampling involves calculating 4 sub points for each pixel.
[superSampling*.tga]

## Reflections
Reflections are performed by casting rays originating from the intersection point directed along the reflection vector.
A portion of the reflected ray's color will be contributed to the original intersection point based on how reflective the surface is.
Fortunately this is fairly easy given the a recursive approach for casting rays.
There is an arbitrary limit on how many reflections a ray can perform before stopping to improve performance and eliminate potential infinite loops.
[reflections.tga]

## Depth of Field
Depth of field is simulated by defining a sharp plane where all objects will be in focus.
The idea is that the camera doesn't detect light at a single precise point.
The final image is composed of light coming from slightly different directions because of the size of the camera.
Any ray passing through the same point on the sharp plane will contribute to the same pixel on the image.
Slightly randomizing the originating point from the camera makes objects not on the focus plane to be out of focus.
Casting many randomized rays and averaging their results creates a blurred effect.
If few rays are cast then the resulting image will be grainy and noisy.
The images below were rendered by casting hundreds of rays per pixel to reduce this effect.
[lineFar.tga]

## Final Renders
Put final renders here.
