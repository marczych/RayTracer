// simplePBO.cpp (Rob Farber)
    
// includes
#include <GL/glut.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
 
// external variables
extern float animTime;
extern unsigned int window_width;
extern unsigned int window_height;
 
// constants (the following should be a const in a header file)
unsigned int image_width = window_width;
unsigned int image_height = window_height;
 
extern "C" void launch_kernel(void* pos, unsigned int, unsigned int, float);
 
// variables
GLuint pbo=0;
GLuint textureID=0;
 
void createPBO(GLuint* pbo)
{
 
  if (pbo) {
    // set up vertex data parameter
    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
     
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}
 
void deletePBO(GLuint* pbo)
{
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
     
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
     
    *pbo = 0;
  }
}
 
void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y)
{
  // Enable Texturing
  glEnable(GL_TEXTURE_2D);
 
  // Generate a texture identifier
  glGenTextures(1,textureID);
 
  // Make this the current texture (remember that GL is state-based)
  glBindTexture( GL_TEXTURE_2D, *textureID);
 
  // Allocate the texture memory. The last parameter is NULL since we only
  // want to allocate memory, not initialize it
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, image_width, image_height, 0,
            GL_BGRA,GL_UNSIGNED_BYTE, NULL);
 
  // Must set the filter mode, GL_LINEAR enables interpolation when scaling
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
  // GL_TEXTURE_2D for improved performance if linear interpolation is
  // not desired. Replace GL_LINEAR with GL_NEAREST in the
  // glTexParameteri() call
 
}
 
void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
 
    *tex = 0;
}
 
void cleanupCuda()
{
  if(pbo) deletePBO(&pbo);
  if(textureID) deleteTexture(&textureID);
}
 
// Run the Cuda part of the computation
void runCuda()
{
  uchar4 *dptr=NULL;
 
  // map OpenGL buffer object for writing from CUDA on a single GPU
  // no data is moved (Win & Linux). When mapped to CUDA, OpenGL
  // should not use this buffer
  cudaGLMapBufferObject((void**)&dptr, pbo);
 
  // execute the kernel
  launch_kernel(dptr, image_width, image_height, animTime);
 
  // unmap buffer object
  cudaGLUnmapBufferObject(pbo);
}
 
void initCuda()
{
  // First initialize OpenGL context, so we can properly set the GL
  // for CUDA.  NVIDIA notes this is necessary in order to achieve
  // optimal performance with OpenGL/CUDA interop.  use command-line
  // specified CUDA device, otherwise use device with highest Gflops/s
  int devCount= 0;
  cudaGetDeviceCount(&devCount);
  if( devCount < 1 )
  {
     printf("No GPUS detected\n");
     exit(EXIT_FAILURE);
  }
  cudaGLSetGLDevice( 0 );
   
  createPBO(&pbo);
  createTexture(&textureID,image_width,image_height);
 
  // Clean up on program exit
  atexit(cleanupCuda);
 
  runCuda();
}
