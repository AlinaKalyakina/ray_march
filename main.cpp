//internal includes
#include "common.h"
#include "ShaderProgram.h"
#include "LiteMath.h"

//External dependencies
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <random>
#include "SOIL/SOIL.h"

static GLsizei WIDTH = 512, HEIGHT = 512; //размеры окна

using namespace LiteMath;

float3 g_camPos(0, 0, 3);//
float3 view_dir(0, 0, -1);
float  cam_rot[2] = {0,0};
int    mx = 0, my = 0;
float speed = 0.1;
const float max_speed = 4;
const float init_speed = 0.1;
const float acceleration = 1.5;
bool soft_shadows = false;
bool fog = false;
//float cam_fov = 45;//


void windowResize(GLFWwindow* window, int width, int height)
{
  WIDTH  = width;
  HEIGHT = height;
}

static void mouseMove(GLFWwindow* window, double xpos, double ypos)
{
  xpos *= 0.05f;
  ypos *= 0.05f;

  int x1 = int(xpos);
  int y1 = int(ypos);

  cam_rot[1] += 0.05f*(x1 - mx);	//Изменение угола поворота
  cam_rot[0] -= 0.05f*(y1 - my);
  view_dir.x = cos(cam_rot[0])*sin(cam_rot[1]);
  view_dir.y = sin(cam_rot[0]);
  view_dir.z = -cos(cam_rot[0])*cos(cam_rot[1]);

  mx = int(xpos);
  my = int(ypos);
}

static void mouseScroll(GLFWwindow* window, double xpos, double ypos)
{
    g_camPos += ypos*view_dir*0.1;
//    if(cam_fov <= 1.0f)
//        cam_fov = 1.0f;
//    if(cam_fov >= 45.0f)
//        cam_fov = 45.0f;
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    switch (key) {
        case GLFW_KEY_W:
            g_camPos += speed*view_dir;
            break;
        case GLFW_KEY_S:
            g_camPos -= speed*view_dir;
            break;
        case GLFW_KEY_D:
            g_camPos -= speed*float3(view_dir.z, 0, -view_dir.x);
            break;
        case GLFW_KEY_A:
            g_camPos += speed*float3(view_dir.z, 0, -view_dir.x);
            break;
        case GLFW_KEY_Q:
            cam_rot[1] -= speed;
            break;
        case GLFW_KEY_E:
            cam_rot[1]+= speed;
            break;
        case GLFW_KEY_R:
            cam_rot[0] += speed;
            break;
        case GLFW_KEY_F:
            cam_rot[0] -= speed;
            break;
        case GLFW_KEY_0:
            cam_rot[0] = cam_rot[1] = 0;
            g_camPos = float3(0, 0, 3);
            break;
        case GLFW_KEY_T:
            g_camPos.y += speed;
            break;
        case GLFW_KEY_G:
            g_camPos.y -= speed;
            break;
        case GLFW_KEY_1:
            if (action == GLFW_PRESS)
                soft_shadows = ! soft_shadows;
            break;
        case GLFW_KEY_2:
            if (action == GLFW_PRESS)
                fog = ! fog;
            break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            if (speed < max_speed && action == GLFW_PRESS) {
                speed *= 2;
            }
            if (action == GLFW_RELEASE)
                speed = 0.1;
            printf("%lf ", speed);
            break;
    }
    view_dir.x = cos(cam_rot[0])*sin(cam_rot[1]);
    view_dir.y = sin(cam_rot[0]);
    view_dir.z = -cos(cam_rot[0])*cos(cam_rot[1]);
}

int initGL()//
{
	int res = 0;
	//грузим функции opengl через glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	std::cout << "Vendor: "   << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: "  << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL: "     << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	return 0;
}

int main(int argc, char** argv)
{
	if(!glfwInit())
    return -1;
//
	//запрашиваем контекст opengl версии 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); 
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE); 

  GLFWwindow*  window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL ray marching sample", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
    double x, y;
	glfwGetCursorPos(window, &x, &y);
	mx = int(0.05*x);
	my = int(0.05*y);
    //glfwSetCursorPosCallback (window, mouseMove);
    glfwSetScrollCallback(window, mouseScroll);
    glfwSetWindowSizeCallback(window, windowResize);
    glfwSetKeyCallback(window, keyCallback);

	glfwMakeContextCurrent(window); 
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

	if(initGL() != 0) 
		return -1;
	
  //Reset any OpenGL errors which could be present for some reason
	GLenum gl_error = glGetError();
	while (gl_error != GL_NO_ERROR)
		gl_error = glGetError();

	//создание шейдерной программы из двух файлов с исходниками шейдеров
	//используется класс-обертка ShaderProgram
	std::unordered_map<GLenum, std::string> shaders;
	shaders[GL_VERTEX_SHADER]   = "../shaders/vertex.glsl";
	shaders[GL_FRAGMENT_SHADER] = "../shaders/fragment.glsl";
	ShaderProgram program(shaders); GL_CHECK_ERRORS;//

  glfwSwapInterval(1); // force 60 frames per second
  
  //Создаем и загружаем геометрию поверхности
  //
  GLuint g_vertexBufferObject;
  GLuint g_vertexArrayObject;//
  //
 
    float quadPos[] =
    {
      -1.0f,  1.0f,	// v0 - top left corner
      -1.0f, -1.0f,	// v1 - bottom left corner
      1.0f,  1.0f,	// v2 - top right corner
      1.0f, -1.0f	  // v3 - bottom right corner
    };
    int width, height;
    unsigned char* image = SOIL_load_image("../index.jpeg", &width, &height, 0, SOIL_LOAD_RGB);
    GLuint texture;
    glGenTextures(1, &texture); GL_CHECK_ERRORS;
    glBindTexture(GL_TEXTURE_2D, texture); GL_CHECK_ERRORS;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); GL_CHECK_ERRORS;
    glGenerateMipmap(GL_TEXTURE_2D);
    SOIL_free_image_data(image);

    g_vertexBufferObject = 0;
    GLuint vertexLocation = 0; // simple layout, assume have only positions at location = 0

    glGenBuffers(1, &g_vertexBufferObject);                                                        GL_CHECK_ERRORS;
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferObject);                                           GL_CHECK_ERRORS;
    glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), (GLfloat*)quadPos, GL_STATIC_DRAW);     GL_CHECK_ERRORS;

    glGenVertexArrays(1, &g_vertexArrayObject);                                                    GL_CHECK_ERRORS;
    glBindVertexArray(g_vertexArrayObject);                                                        GL_CHECK_ERRORS;

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferObject);                                           GL_CHECK_ERRORS;
    glEnableVertexAttribArray(vertexLocation);                                                     GL_CHECK_ERRORS;
    glVertexAttribPointer(vertexLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);                            GL_CHECK_ERRORS;

    glBindVertexArray(0);


	//цикл обработки сообщений и отрисовки сцены каждый кадр
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		//очищаем экран каждый кадр
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);               GL_CHECK_ERRORS;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); GL_CHECK_ERRORS;

        program.StartUseShader();                          GL_CHECK_ERRORS;

        float4x4 camRotMatrix   = mul(rotate_Y_4x4(-cam_rot[1]), rotate_X_4x4(+cam_rot[0]));
        float4x4 camTransMatrix = translate4x4(g_camPos);
        float4x4 rayMatrix      = mul(camTransMatrix, camRotMatrix);
        program.SetUniform("g_rayMatrix", rayMatrix);
        program.SetUniform("show_fog", fog);
        program.SetUniform("g_screenWidth" , WIDTH);
        program.SetUniform("g_screenHeight", HEIGHT);
        program.SetUniform("show_soft_shadows", soft_shadows);
        // очистка и заполнение экрана цвет//
        glViewport  (0, 0, WIDTH, HEIGHT);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // draw calli
    //
    glBindVertexArray(g_vertexArrayObject); GL_CHECK_ERRORS;
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  GL_CHECK_ERRORS;  // The last parameter of glDrawArrays is equal to VS invocations
    
    program.StopUseShader();

		glfwSwapBuffers(window); 
	}

	//очищаем vboи vao перед закрытием программы
  //
	glDeleteVertexArrays(1, &g_vertexArrayObject);
  glDeleteBuffers(1,      &g_vertexBufferObject);

	glfwTerminate();
	return 0;
}
