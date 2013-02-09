//callbacksPBO.cpp (Rob Farber)

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include "RayTracer.h"

const double PI = atan(1.0)*4;
// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;
int m_startX;
int m_startY;
float m_angleX;
float m_angleY;

void calcLookAtPosition();
float degreesToRadians(float);

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern RayTracer* g_rayTracer;
extern void moveIn();
extern void moveOut();
extern void moveUp();
extern void moveDown();
extern void moveLeft();
extern void moveRight();

// The user must create the following routines:
void runCuda();

void display()
{
   // run CUDA kernel
   runCuda();

   // Create a texture from the buffer
   glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

   // bind texture from PBO
   glBindTexture(GL_TEXTURE_2D, textureID);


   // Note: glTexSubImage2D will perform a format conversion if the
   // buffer is a different format from the texture. We created the
   // texture with format GL_RGBA8. In glTexSubImage2D we specified
   // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

   // Note: NULL indicates the data resides in device memory
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_rayTracer->width, g_rayTracer->height,
         GL_RGBA, GL_UNSIGNED_BYTE, NULL);


   // Draw a single Quad with texture coordinates for each vertex.

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   // Don't forget to swap the buffers!
   glutSwapBuffers();

   // if animFlag is true, then indicate the display needs to be redrawn
   if(animFlag) {
      glutPostRedisplay();
      animTime += animInc;
   }
}

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
   switch(key) {
   case 'w':
      g_rayTracer->camera.position -= g_rayTracer->camera.w;
      g_rayTracer->camera.lookAt -= g_rayTracer->camera.w;
      break;
   case 's':
      g_rayTracer->camera.position += g_rayTracer->camera.w;
      g_rayTracer->camera.lookAt += g_rayTracer->camera.w;
      break;

      // Strafe movement.
   case 'a':
      g_rayTracer->camera.position -= g_rayTracer->camera.v;
      g_rayTracer->camera.lookAt -= g_rayTracer->camera.v;
      break;
   case 'd':
      g_rayTracer->camera.position += g_rayTracer->camera.v;
      g_rayTracer->camera.lookAt += g_rayTracer->camera.v;
      break;
   case(27) :
      exit(0);
      break;
   }

   g_rayTracer->camera.calculateWUV();

   // indicate the display must be redrawn
   glutPostRedisplay();
}

// No mouse event handlers defined
void mouse(int button, int state, int x, int y) {
   if (button != GLUT_LEFT_BUTTON) {
      return;
   }

   bool down = state == GLUT_DOWN;
   y = g_rayTracer->height - y - 1;

   if (down) {
      m_startX = x;
      m_startY = y;
   }

   glutPostRedisplay();
}

void motion(int x, int y) {
   y = g_rayTracer->height - y - 1;

   int diffX = m_startX - x;
   int diffY = m_startY - y;

   /* One to one pixels for degrees. */
   float angleChangeX = diffX * 180.0f / g_rayTracer->width;
   float angleChangeY = diffY * 180.0f / g_rayTracer->height;

   m_angleX += angleChangeX;
   m_angleY = fmin(fmax(m_angleY + angleChangeY, -50.0f), 50.0f);

   m_startX = x;
   m_startY = y;

   calcLookAtPosition();
   glutPostRedisplay();
}

void calcLookAtPosition() {
   float theta = degreesToRadians(-m_angleX);
   float phi = degreesToRadians(-m_angleY);

   g_rayTracer->camera.lookAt.x = sin(phi);
   g_rayTracer->camera.lookAt.y = cos(phi) * cos(theta);
   g_rayTracer->camera.lookAt.z = cos(phi) * cos(degreesToRadians(90) - theta);

   g_rayTracer->camera.lookAt = g_rayTracer->camera.lookAt.normalize() *
    g_rayTracer->camera.lookAtLength + g_rayTracer->camera.position;
}

float degreesToRadians(float degrees) {
   return degrees * PI / 180;
}
