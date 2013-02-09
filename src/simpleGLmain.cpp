// simpleGLmain.cpp (Rob Farber)

// includes
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

// The user must create the following routines:
// CUDA methods
extern void initCuda();
extern void runCuda();
extern void renderCuda(int);

// callbacks
extern void display();
extern void keyboard(unsigned char key, int x, int y);
extern void mouse(int button, int state, int x, int y);
extern void motion(int x, int y);

// GLUT specific variables
unsigned int window_width = 1024;
unsigned int window_height = 1024;


// Forward declarations of GL functionality
bool initGL(int argc, char** argv);
struct timeval start, end;

// Simple method to display the Frames Per Second in the window title
void computeFPS()
{
   static int fpsCount=0;
   static int fpsLimit=100;

   fpsCount++;

   if (fpsCount == fpsLimit) {
      char fps[256];
      long seconds  = end.tv_sec  - start.tv_sec;
      long useconds = end.tv_usec - start.tv_usec;

      long utime = ((seconds) * 1000000 + useconds);
      double ifps = 1.0 /utime;//1.f / (mtime / 1000.f);
      ifps = ifps * (1000000);// 1/usec -> 1/sec
      sprintf(fps, "Cuda GL Interop Wrapper: %3.4f fps, Total Time: %3.4fms", ifps, (utime/(float)1000));

      glutSetWindowTitle(fps);
      fpsCount = 0;
   }
}

void fpsDisplay()
{
   gettimeofday( &start, NULL );
   display();

   gettimeofday( &end, NULL );
   computeFPS();
}

// Main program
int main(int argc, char** argv)
{
   if( argc > 1 )
   {
      int width = atoi(argv[1]);
      int height = atoi(argv[2]);
      if( width > 100 && height > 100 )
      {
         window_width = width;
         window_height = height;
      }
   }
   if (false == initGL(argc, argv)) {
      return false;
   }

   initCuda();

   // register callbacks
   glutDisplayFunc(fpsDisplay);
   glutKeyboardFunc(keyboard);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);

   // start rendering mainloop
   glutMainLoop();
}

bool initGL(int argc, char **argv)
{
   //Steps 1-2: create a window and GL context (also register callbacks)
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(window_width, window_height);
   glutCreateWindow("Cuda GL Interop Demo (adapted from NVIDIA's simpleGL");
   glutDisplayFunc(fpsDisplay);
   glutKeyboardFunc(keyboard);
   glutMotionFunc(motion);

   // Step 3: Setup our viewport and viewing modes
   glViewport(0, 0, window_width, window_height);

   glClearColor(0.0, 0.0, 0.0, 1.0);
   glDisable(GL_DEPTH_TEST);


   // set view matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

   return true;
}
